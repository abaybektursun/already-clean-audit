"""V2b — attractor vs scar: long continuation under gauges with periodic stats.

Extends the M-Q2 design (machinery reused verbatim from the mq2 job's imports)
to a long horizon with checkpoints every `ckpt_every` steps. Three pre-named
branches (spec §1): (i) fast re-formation at new coordinates = optimizer
account; (ii) re-formation paced by outlier re-growth = activation-address
account; (iii) no re-formation = historical scar. Instrument-validity gate:
the unrotated control must KEEP its sink (final delivery(core32) < 1), else
the arm is VOID-INSTRUMENT.

Per checkpoint, per (gauge, seed), banked as rows in v2b_rows.jsonl:
  delivery_win  per bank: HORIZON-MATCHED — motion IN THIS WINDOW over gradient
                IN THIS WINDOW (AM-5 instrument fix, review item 92). The banked
                V2b construction divided CUMULATIVE motion (dW from step 0) by
                WINDOWED gradient share, so its numerator and denominator ran
                over different horizons and the ratio drifted with the horizon
                itself. Both are emitted now: delivery_win is primary, delivery
                (cumulative/windowed) is kept verbatim for comparability with
                the banked rows.
  delivery      the banked construction, unchanged
  vhat_share    tr(U^T diag(colsum v) U) / tr(diag(colsum v)) for core32 —
                Adam's second-moment concentration in the transported core
  act_per_layer PER LAYER at q_proj_in on eval batches: max|x_i|, top-coord
                energy share (AM-5 instrument fix (b)). The banked rows kept
                only the MAX over layers, which is attained at layer 0 — whose
                q_proj input is the embedding through a norm and therefore does
                NOT depend on any trained q_proj weight. That is why the banked
                hadamard tracker is bit-identical across checkpoints. The scalar
                maxima are still emitted; act_distinct_max counts distinct
                per-layer values so the defect cannot recur unnoticed.
  fresh_bottom  motion share into the CURRENT weights' bottom-32 right-singular
                subspace, recomputed at the checkpoint (a re-formed sink at new
                coordinates lives here)

DISCOVERY register. FAIL LOUD: no try/except.
"""
import json
import os
import time


def run(cell, out_dir):
    # hf_setup MUST precede any huggingface_hub/transformers import — the hub
    # freezes HF_HUB_OFFLINE into constants at import time (D17, house memory)
    from infra.pg import humboldt_core as H
    H.hf_setup()
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from infra.pg import mercator_gauge as G
    from infra.pg import mercator_arch as AR
    from infra.pg import humboldt_weights as WT
    from infra.pg import mercator_core as M
    from infra.pg import darwin_train as T
    from infra.aws_pg.jobs.invariant_core_func import (_blank_image, _decoder_layers,
                                                       _ppl_blankimg)
    from infra.aws_pg.jobs.darwin_w1_u import build_batches, eval_task_loss, text_task_rows
    from infra.aws_pg.jobs.mercator_mq import (_layers, _reference_logits, _untie_lm_head,
                                               fold_model, rotate_model, scale_gauge_model)
    from infra.aws_pg.jobs.mercator_mq2 import orth, MOTION_BANKS, BANK
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    gauges = cell["gauges"].split(",")
    seeds = [int(s) for s in cell["seeds"].split(",")]
    steps, ckpt = int(cell["steps"]), int(cell["ckpt_every"])
    lr = float(cell.get("lr", 1e-5))

    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    blank = _blank_image(128)
    gate_texts = ["The capital of France is Paris, and the capital of Italy is",
                  "In 1969 the first humans landed on the",
                  "Water boils at one hundred degrees",
                  "The quick brown fox jumps over the lazy"]
    train_rows, eval_rows = text_task_rows(int(cell.get("n_train", 200)),
                                           int(cell.get("n_eval", 32)), seeds[0])
    bank_dir = M.fetch_all_banks(H.cache_dir(out_dir, "banks"))
    layers_idx = list(range(H.N_LAYERS))
    raw = {L: {b: WT.load_bank(bank_dir, "q_proj", L, b) for b in MOTION_BANKS}
           for L in layers_idx}

    def load():
        m = PaliGemmaForConditionalGeneration.from_pretrained(
            WT.PG1_REPO, revision=base_sha, torch_dtype=torch.float32,
            device_map="cuda", attn_implementation="eager")
        _untie_lm_head(m)
        return m

    model = load()
    model.eval()
    ref_logits = _reference_logits(model, processor, "cuda", gate_texts, blank)
    _lm, layers = _layers(model)
    gains = {L: G.gemma_effective_gain(
        layers[L].input_layernorm.weight.detach().cpu().numpy()) for L in layers_idx}
    n_scaled = scale_gauge_model(model, G.REASSOCIATION_PROBE_SCALE)
    floor = G.function_preservation(
        ref_logits, _reference_logits(model, processor, "cuda", gate_texts, blank))
    del model, _lm, layers
    torch.cuda.empty_cache()

    folded = {L: {b: orth(gains[L][:, None] * raw[L][b]) for b in MOTION_BANKS}
              for L in layers_idx}
    d = folded[0][BANK].shape[0]

    rows_f = open(os.path.join(out_dir, "v2b_rows.jsonl"), "w")
    gates = {}

    def emit(row):
        rows_f.write(json.dumps(row) + "\n")
        rows_f.flush()

    def act_stats(model, mods, eval_b):
        """per-layer q_proj_in outlier trackers on 2 eval batches."""
        acc = {}
        hooks = []
        def mk(L):
            def hook(module, args):
                h = args[0].detach().to(torch.float64)
                a = acc.setdefault(L, {"mx": 0.0,
                                       "col2": torch.zeros(h.shape[-1], dtype=torch.float64,
                                                           device=h.device)})
                a["col2"] += h.pow(2).sum(dim=tuple(range(h.dim() - 1)))
                a["mx"] = max(a["mx"], float(h.abs().max()))
            return hook
        for (t, L), m in mods.items():
            hooks.append(m.register_forward_pre_hook(mk(L)))
        with torch.no_grad():
            for b in eval_b[:2]:
                model(input_ids=b["input_ids"], labels=b["labels"])
        for h in hooks:
            h.remove()
        out = {}
        for L, a in acc.items():
            col = a["col2"]
            out[L] = {"max_absx": a["mx"],
                      "top_coord_energy_share": float(col.max() / col.sum())}
        return out

    for gname in gauges:
        model = load()
        model.eval()
        fold_model(model)
        R = None
        if gname == "hadamard":
            R = G.hadamard_rotation(d)
        elif gname != "unrotated":
            R = AR.random_rotation(d, seed=int(gname.split("_s")[1]))
        if R is not None:
            rotate_model(model, R)
        fp = G.function_preservation(
            ref_logits, _reference_logits(model, processor, "cuda", gate_texts, blank))
        fp_gate = G.assert_function_preserved(fp, what=f"gauge:{gname}",
                                              noise_floor=floor["max_abs_diff"])
        gates[f"fp_{gname}"] = {"passed": True, **{k: v for k, v in fp.items()
                                                   if isinstance(v, (int, float))}}

        U = {L: {b: torch.as_tensor(folded[L][b] if R is None else orth(R @ folded[L][b]),
                                    device="cuda", dtype=torch.float32)
                 for b in MOTION_BANKS} for L in layers_idx}
        for p in model.parameters():
            p.requires_grad_(False)
        mods = T.trainable_modules(_decoder_layers(model), types=("q_proj",),
                                   layers=layers_idx)
        for m in mods.values():
            m.weight.requires_grad_(True)
        pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
        train_b = build_batches(processor, train_rows, "cuda")
        eval_b = build_batches(processor, eval_rows, "cuda")

        for seed in seeds:
            for k, m in mods.items():
                m.weight.data.copy_(pristine[k])
            opt = T.ExactAdam([m.weight for m in mods.values()], lr=lr)
            # horizon-matched delivery needs the weights AT THE LAST CHECKPOINT,
            # not only the pristine ones
            last_ckpt_W = {k: v.clone() for k, v in pristine.items()}
            rng = np.random.default_rng(H.cell_seed(seed, "V2B", gname, "cont"))
            gwin = {b: 0.0 for b in MOTION_BANKS}
            gwin_tot = 0.0
            model.train()
            t0 = time.time()
            for step in range(1, steps + 1):
                batch = train_b[int(rng.integers(0, len(train_b)))]
                opt.zero_grad()
                model(input_ids=batch["input_ids"], labels=batch["labels"]).loss.backward()
                with torch.no_grad():
                    for (t, L), m in mods.items():
                        g = m.weight.grad.to(torch.float32)
                        gwin_tot += float((g * g).sum())
                        for b in MOTION_BANKS:
                            gwin[b] += float((g @ U[L][b]).pow(2).sum())
                opt.step()
                if step % ckpt == 0 or step == steps:
                    model.eval()
                    with torch.no_grad():
                        mtot, mins = 0.0, {b: 0.0 for b in MOTION_BANKS}
                        wtot, wins = 0.0, {b: 0.0 for b in MOTION_BANKS}
                        fresh_in = fresh_in_win = 0.0
                        vh_num, vh_den = 0.0, 0.0
                        for idx, ((t, L), m) in enumerate(mods.items()):
                            W = opt.master[idx]
                            dW = W - pristine[(t, L)]
                            dWw = W - last_ckpt_W[(t, L)]
                            last_ckpt_W[(t, L)] = W.clone()
                            mtot += float((dW * dW).sum())
                            wtot += float((dWw * dWw).sum())
                            for b in MOTION_BANKS:
                                mins[b] += float((dW @ U[L][b]).pow(2).sum())
                                wins[b] += float((dWw @ U[L][b]).pow(2).sum())
                            # fresh bottom-32 right-singular subspace of CURRENT W
                            _, _, Vh = torch.linalg.svd(W, full_matrices=False)
                            Ub = Vh[-32:, :].T
                            fresh_in += float((dW @ Ub).pow(2).sum())
                            fresh_in_win += float((dWw @ Ub).pow(2).sum())
                            colv = opt.v[idx].sum(dim=0)
                            Uc = U[L][BANK]
                            vh_num += float((colv @ (Uc * Uc).sum(dim=1)))
                            vh_den += float(colv.sum())
                        acts = act_stats(model, mods, eval_b)
                        ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=4)["ppl"])
                        task = eval_task_loss(model, eval_b)
                    row = {"gauge": gname, "seed": seed, "step": step,
                           "elapsed_s": round(time.time() - t0, 1),
                           "delivery": {b: (mins[b] / mtot) / (gwin[b] / gwin_tot)
                                        for b in MOTION_BANKS},
                           "delivery_win": {b: (wins[b] / wtot) / (gwin[b] / gwin_tot)
                                            for b in MOTION_BANKS},
                           "grad_share_win": {b: gwin[b] / gwin_tot for b in MOTION_BANKS},
                           "motion_share": {b: mins[b] / mtot for b in MOTION_BANKS},
                           "motion_share_win": {b: wins[b] / wtot for b in MOTION_BANKS},
                           "fresh_bottom32_motion_share": fresh_in / mtot,
                           "fresh_bottom32_motion_share_win": fresh_in_win / wtot,
                           "vhat_core_share": vh_num / vh_den,
                           "act_max_absx": max(a["max_absx"] for a in acts.values()),
                           "act_top_coord_share_max": max(a["top_coord_energy_share"]
                                                          for a in acts.values()),
                           "act_per_layer": {str(L): a for L, a in sorted(acts.items())},
                           "act_distinct_max": len({round(a["max_absx"], 6)
                                                    for a in acts.values()}),
                           "ppl": ppl, "task_loss": task}
                    emit(row)
                    print(f"[v2b] {gname}/s{seed} step {step}: "
                          f"delivery(core)={row['delivery'][BANK]:.3f} "
                          f"fresh_bottom={row['fresh_bottom32_motion_share']:.4f} "
                          f"vhat_core={row['vhat_core_share']:.4f} task={task:.4f}")
                    gwin = {b: 0.0 for b in MOTION_BANKS}
                    gwin_tot = 0.0
                    model.train()
            model.eval()
        del model, U, mods, pristine
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    rows_f.close()
    with open(os.path.join(out_dir, "gates.json"), "w") as f:
        json.dump(gates, f, indent=2)
    with open(os.path.join(out_dir, "v2b_config.json"), "w") as f:
        json.dump({"gauges": gauges, "seeds": seeds, "steps": steps,
                   "ckpt_every": ckpt, "lr": lr, "base_sha": base_sha,
                   "noise_floor": floor["max_abs_diff"]}, f, indent=2)
