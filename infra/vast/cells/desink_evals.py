"""AM-7 — does de-sinking buy any actual task capability?

V2a established that fine-tuning under a Hadamard gauge lowers the TRAINING-task
loss relative to unrotated at matched seeds. That is the trained objective. This
cell asks whether anything a person would call a capability moves with it, at
matched steps AND at matched wall-clock, on benchmarks the fine-tune never saw.

  BENCHMARKS (frozen at freeze from what the baked harness actually supports on
  PG1's in-distribution blank-image forward — both bank per_item in a fixed item
  order, so arms are paired BY POSITION, never by the colliding item hash):
    mmlu_blankimg  5-shot MMLU, single-token answer-letter argmax (recognition)
    mmlu_cloze     5-shot MMLU, rank the four choice TEXTS (content axis)
  Kept alongside as continuity readouts, NOT as the benchmark family: the
  blank-image wikitext ppl and the held-out training-task loss.

  MATCHED WALL-CLOCK. Rotation costs a one-time fold+rotate+gate setup and (as
  measured, not assumed) essentially nothing per step. The honest matched
  comparison therefore charges the setup: the reference gauge runs first and
  fixes a total wall-clock budget (setup + training); every other gauge is
  additionally evaluated at the last step that fits inside that budget. Steps,
  seconds, and the setup share are all banked, so the reader can see how much of
  the "cost" is the intervention and how much is per-step arithmetic.

Everything else (fold, rotation, gauge gate, ExactAdam, banks, task rows) is the
M-Q2 machinery, imported verbatim. FAIL LOUD.
"""
import json
import os
import time


def run(cell, out_dir):
    from infra.pg import humboldt_core as H
    H.hf_setup()                                  # BEFORE any hub import (D17)
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from infra.pg import mercator_gauge as G
    from infra.pg import mercator_arch as AR
    from infra.pg import humboldt_weights as WT
    from infra.pg import mercator_core as M
    from infra.pg import darwin_train as T
    from infra.pg import text_eval as TE
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

    gauges = cell["gauges"].split(",")             # reference gauge FIRST
    seeds = [int(s) for s in cell["seeds"].split(",")]
    steps = int(cell["steps"])
    lrs = cell["lrs"]                              # {gauge: lr} — AM-2's best-of
    assert set(lrs) >= set(gauges), (lrs, gauges)
    eval_steps = [int(s) for s in cell.get("eval_steps", [120, steps])]
    n_per_subject = int(cell.get("n_per_subject", 8))
    n_seq = int(cell.get("n_seq", 8))

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
    scale_gauge_model(model, G.REASSOCIATION_PROBE_SCALE)
    floor = G.function_preservation(
        ref_logits, _reference_logits(model, processor, "cuda", gate_texts, blank))
    del model, _lm, layers
    torch.cuda.empty_cache()

    folded = {L: {b: orth(gains[L][:, None] * raw[L][b]) for b in MOTION_BANKS}
              for L in layers_idx}
    d = folded[0][BANK].shape[0]

    def benchmarks(model, tag):
        t0 = time.time()
        mm = TE.mmlu_blankimg(model, processor, "cuda", n_shots=5,
                              n_per_subject=n_per_subject)
        cz = TE.mmlu_cloze(model, processor.tokenizer, "cuda", processor=processor,
                           n_shots=5, n_per_subject=n_per_subject)
        return {"mmlu_blankimg": {"acc": mm["acc"], "n": mm["n"],
                                  "per_item": mm["per_item"]},
                "mmlu_cloze": {"acc": cz["acc"], "n": cz["n"],
                               "per_item": cz["per_item"]},
                "bench_seconds": round(time.time() - t0, 1), "tag": tag}

    rows_f = open(os.path.join(out_dir, "am7_rows.jsonl"), "w")

    def emit(row):
        rows_f.write(json.dumps(row) + "\n")
        rows_f.flush()

    budget_s = None            # total wall-clock (setup + train) of the reference
    gates = {}
    for gi, gname in enumerate(gauges):
        t_setup = time.time()
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
        G.assert_function_preserved(fp, what=f"gauge:{gname}",
                                    noise_floor=floor["max_abs_diff"])
        gates[f"fp_{gname}"] = {k: v for k, v in fp.items()
                                if isinstance(v, (int, float))}
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
        setup_s = time.time() - t_setup
        print(f"[am7] {gname}: setup {setup_s:.0f}s, lr={lrs[gname]}", flush=True)

        # step-0 (base) benchmarks once, on the gauge itself — the gauge is a
        # function-preserving rewrite, so this is also a check on that claim
        emit({"gauge": gname, "seed": None, "step": 0, "kind": "base",
              "setup_s": round(setup_s, 1), "lr": lrs[gname],
              **benchmarks(model, "base"),
              "ppl": float(_ppl_blankimg(model, processor, "cuda", n_seq=n_seq)["ppl"]),
              "task_loss": float(eval_task_loss(model, eval_b))})

        for seed in seeds:
            for k, m in mods.items():
                m.weight.data.copy_(pristine[k])
            opt = T.ExactAdam([m.weight for m in mods.values()], lr=float(lrs[gname]))
            rng = np.random.default_rng(H.cell_seed(seed, "AM7", gname, "evals"))
            model.train()
            t0 = time.time()
            train_s = eval_s = 0.0
            # the reference gauge SETS the budget, so it never needs a
            # wall-clock-matched row of its own
            wall_done = (gi == 0)
            todo = sorted(set(eval_steps))
            for step in range(1, steps + 1):
                batch = train_b[int(rng.integers(0, len(train_b)))]
                opt.zero_grad()
                model(input_ids=batch["input_ids"], labels=batch["labels"]).loss.backward()
                opt.step()
                # evaluation time is not training time: the wall-clock axis
                # compares TRAINING cost, so eval seconds are subtracted out
                train_s = time.time() - t0 - eval_s
                hit = [s for s in todo if s == step]
                wall_hit = (not wall_done) and (setup_s + train_s >= budget_s)
                if not hit and not wall_hit:
                    continue
                model.eval()
                t_eval = time.time()
                kinds = (["step"] if hit else []) + (["wallclock"] if wall_hit else [])
                res = {**benchmarks(model, f"{gname}/s{seed}@{step}"),
                       "ppl": float(_ppl_blankimg(model, processor, "cuda",
                                                  n_seq=n_seq)["ppl"]),
                       "task_loss": float(eval_task_loss(model, eval_b))}
                for kind in kinds:
                    emit({"gauge": gname, "seed": seed, "step": step, "kind": kind,
                          "train_s": round(train_s, 1), "setup_s": round(setup_s, 1),
                          "total_s": round(setup_s + train_s, 1),
                          "budget_s": budget_s, "lr": lrs[gname], **res})
                    print(f"[am7] {gname}/s{seed} step {step} ({kind}): "
                          f"mmlu={res['mmlu_blankimg']['acc']:.4f} "
                          f"cloze={res['mmlu_cloze']['acc']:.4f} "
                          f"ppl={res['ppl']:.3f} task={res['task_loss']:.4f}",
                          flush=True)
                if wall_hit:
                    wall_done = True
                eval_s += time.time() - t_eval
                model.train()
            model.eval()
            if budget_s is None:                 # first seed of the reference
                budget_s = setup_s + train_s
                print(f"[am7] wall-clock budget set by {gname}: {budget_s:.0f}s",
                      flush=True)
        del model, U, mods, pristine
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    rows_f.close()
    with open(os.path.join(out_dir, "am7_config.json"), "w") as f:
        json.dump({"gauges": gauges, "seeds": seeds, "steps": steps, "lrs": lrs,
                   "eval_steps": eval_steps, "n_per_subject": n_per_subject,
                   "budget_s": budget_s, "base_sha": base_sha,
                   "noise_floor": floor["max_abs_diff"], "gates": gates,
                   "benchmarks": ["mmlu_blankimg", "mmlu_cloze"],
                   "pairing": "by position (identical item order); never by item hash"},
                  f, indent=2)
