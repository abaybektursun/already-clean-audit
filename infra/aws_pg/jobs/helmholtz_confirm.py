"""HELMHOLTZ confirmatory arm — the MMLU-targeted crosscoder-blindness job.

For a (before, after) text-checkpoint pair it measures, on the SAME paired residual-stream
tokens at one layer, whether a model-diff reports the representation as PRESERVED while the
model's MMLU capability is destroyed — and whether the diff is SENSITIVE (positive control)
so "preserved" is meaningful rather than coarse. Concretely it computes, all on the same pair:

  1. linear_cka(before, after)                        — the representation-similarity signal.
  2. a crosscoder trained on the REAL pair            — {survival, lost, emergent} + FVU.
  3. a crosscoder trained on a PLANTED pair           — the positive control; it MUST register
     (diff_metrics.plant_direction on the after side)   the plant as MOVED, else H_coarse/VOID.
  4. mmlu_accuracy + mmlu_cloze on the AFTER model    — the behavioural readouts.

Everything downstream of the harvest is the frozen pure core (infra/pg/helmholtz_core.py):
linear_cka, preservation_verdict, positive_control_ok, adjudicate.

SMOKE (wiring validation): --before-repo google/gemma-2-2b --after-repo google/gemma-2-2b
(an IDENTITY pair). Expected wiring-valid signal: CKA ~1.0; the real crosscoder PRESERVED
inside the identity band (survival ~1, tiny lost/emergent — forced by before==after symmetry);
the plant control FIRES (planted pair MOVED); mmlu runs and returns a number.

Fail-loud (CLAUDE.md / aws-pg skill): asserts, no try/except swallowing. Heavy imports inside
main() so the module py_compiles on the control plane without torch.

  python -m infra.aws_pg.jobs.helmholtz_confirm --out <dir> \
      --before-repo google/gemma-2-2b --after-repo google/gemma-2-2b \
      --layer 12 --n-per-subject 2 --smoke
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

# Identity-pair tolerance band the crosscoder verdict is read against. In the smoke the REAL
# pair IS the identity pair (base vs base), whose lost/emergent are ~0 by before==after symmetry,
# so a small fixed band lands it PRESERVED while the (large) plant clears it and reads MOVED. The
# confirmatory science arm derives this band from a dedicated base-vs-base identity crosscoder; a
# fixed band is a wiring device only and is recorded verbatim in metrics.json.
IDENTITY_BAND = {"lost": 0.02, "emergent": 0.02}


def _harvest_texts(n_per_subject):
    """The paired-harvest corpus: MMLU test items rendered as `_q_block` prompts (question +
    4 choices, ending at 'Answer:'), n_per_subject per subject in canonical sorted-subject order.
    Same content family as the behavioural readouts, so the crosscoder sees the residuals over the
    very items whose MMLU behaviour is measured."""
    from collections import defaultdict
    from datasets import load_dataset
    from infra.pg.text_eval import MMLU_REPO, _q_block

    ds = load_dataset(MMLU_REPO, "all")
    by_subj = defaultdict(list)
    for r in ds["test"]:
        by_subj[r["subject"]].append(r)
    texts = []
    for subj in sorted(by_subj):
        rows = by_subj[subj][:n_per_subject] if n_per_subject else by_subj[subj]
        for r in rows:
            texts.append(_q_block(r["question"], r["choices"]))
    assert texts, "harvest corpus empty"
    return texts


def harvest_paired_text(m_before, m_after, layer_idx, texts, tokenizer, device, micro=8):
    """Forward both models over the SAME text tokens, hook the decoder layer `layer_idx` residual
    on each, keep real (attention_mask) positions. Returns (acts_before, acts_after) as float32
    CPU tensors row-aligned by token position (token i is the same token in both models).

    Reuses the validated harvest primitives (ResidualHarvest, resolve_gemma2_layers); the paired
    multimodal `diff_harvest.harvest_paired_layer` cannot be reused verbatim here because gemma-2-2b
    is a text-only causal LM (no pixel_values)."""
    import torch
    from infra.pg.gemma_scope import ResidualHarvest, resolve_gemma2_layers

    lb = resolve_gemma2_layers(m_before)
    la = resolve_gemma2_layers(m_after)
    assert layer_idx < len(lb) and layer_idx < len(la), \
        f"layer {layer_idx} out of range (before {len(lb)} / after {len(la)} layers)"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bef, aft = [], []
    with ResidualHarvest(lb, layer_idx) as hb, ResidualHarvest(la, layer_idx) as ha:
        for s in range(0, len(texts), micro):
            enc = tokenizer(texts[s:s + micro], return_tensors="pt", padding=True,
                            add_special_tokens=True).to(device)
            km = enc["attention_mask"].reshape(-1).bool()
            with torch.inference_mode():
                m_before(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            ab = hb.captured.reshape(-1, hb.captured.shape[-1])[km].float().cpu()
            with torch.inference_mode():
                m_after(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            aa = ha.captured.reshape(-1, ha.captured.shape[-1])[km].float().cpu()
            assert ab.shape == aa.shape, f"paired shape mismatch L{layer_idx}: {ab.shape} vs {aa.shape}"
            bef.append(ab)
            aft.append(aa)
    acts_before = torch.cat(bef, 0)
    acts_after = torch.cat(aft, 0)
    assert acts_before.shape == acts_after.shape and acts_before.ndim == 2
    return acts_before, acts_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--before-repo", default="google/gemma-2-2b")
    ap.add_argument("--after-repo", default="google/gemma-2-2b")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-per-subject", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true", help="capped crosscoder steps; wiring only")
    ap.add_argument("--real", action="store_true",
                    help="confirmatory science: source the skill-loss pair via "
                         "diff_harvest.build_before_after (AFTER=PaliGemma-2's install-deleted LM, "
                         "BEFORE=base gemma-2-2b LM), extract the Gemma2 language_model submodules, "
                         "score MMLU on BOTH sides. CKA-plant is the positive control; the "
                         "crosscoder counter is corroboration only (its plant control is void).")
    ap.add_argument("--successor", action="store_true",
                    help="readout-only damage (prereg 2026-09-01_helmholtz_successor_prereg.md): "
                         "BEFORE=frozen base gemma-2-2b, AFTER=base + LoRA trained on PERMUTED-label "
                         "MMLU items (relearn.shuffle_labels) — scrambles the question->letter "
                         "binding without content change. Frozen recipe: rank=16 alpha=16 n=2000 "
                         "lr=5e-6 1 epoch batch=8. Prediction: MCQ drop>0.5, cloze drop<0.2, CKA>0.9.")
    a = ap.parse_args()
    assert not (a.real and a.successor), "--real and --successor are mutually exclusive"
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # gated gemma-2-2b + MMLU streaming both need the SSM HF token + online mode (run_job.sh sets
    # HF_HUB_OFFLINE=1). Call FIRST, before transformers/datasets import.
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from infra.pg import diff_metrics as DM
    from infra.pg import helmholtz_core as HC
    from infra.pg import tessera_crosscoder as XC
    from infra.pg import text_eval as TE

    assert torch.cuda.is_available(), "HELMHOLTZ confirm needs a GPU"
    device = "cuda"
    steps = 1000 if a.smoke else XC.CC_TRAIN_STEPS
    rep = {"object": "HELMHOLTZ confirmatory — MMLU crosscoder-blindness",
           "smoke": a.smoke, "before_repo": a.before_repo, "after_repo": a.after_repo,
           "layer": a.layer, "n_per_subject": a.n_per_subject, "seed": a.seed,
           "identity_band": IDENTITY_BAND, "crosscoder_steps": steps,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    # ---- load the before/after pair (Gemma2 -> eager attention, matches diff_harvest) ----
    if a.real:
        # Confirmatory science: the skill-loss pair via build_before_after (eager for the prefix-LM
        # readout). AFTER = PaliGemma-2 as released (its LM's text MMLU-MCQ is deleted per REPRICE);
        # BEFORE = same shell with the LM reset to base gemma-2-2b. Extract the Gemma2 language_model
        # submodules -> both are plain causal LMs, so the validated text harvest + mmlu paths apply
        # unchanged (text-only input; no image tokens, so the prefix-LM mask subtlety does not arise).
        from infra.pg.diff_harvest import build_before_after
        print("[HELM] --real: build_before_after (PG2 install-deleted LM vs base LM)", flush=True)
        pg_after, pg_before, proc, paths, _iti, _gate = build_before_after(device, pg_key="PG",
                                                                          attn_impl="eager")
        lm_path = paths["language_model"]
        m_after = pg_after.get_submodule(lm_path).eval()
        m_before = pg_before.get_submodule(lm_path).eval()
        tok = proc.tokenizer
        rep["before_repo"] = "build_before_after:base-gemma-2-2b-LM"
        rep["after_repo"] = "build_before_after:PG2-install-deleted-LM"
    elif a.successor:
        # Readout-only damage (frozen prereg recipe): BEFORE = frozen base gemma-2-2b; AFTER = a
        # second copy + LoRA trained on PERMUTED-label MMLU auxiliary_train items — single-letter CE
        # toward WRONG letters, computed exactly like the eval read, so it scrambles the
        # question->letter binding while the training text carries no content change.
        from datasets import load_dataset
        from infra.pg import relearn as Rl
        print("[HELM] --successor: base vs base+shuffled-label-LoRA (readout-only damage)", flush=True)
        m_before = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b", torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager").eval()
        m_after = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b", torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager").eval()
        tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        RECIPE = {"rank": 16, "alpha": 16.0, "n": 2000, "lr": 5e-6, "epochs": 1,
                  "batch": 8, "warmup": 0.03}
        ds = load_dataset(TE.MMLU_REPO, "all")
        pool = [dict(r) for r in ds["auxiliary_train"]]
        ev_hashes = [Rl.qa_item_hash(r["question"]) for r in ds["test"]]
        ded = Rl.dedup_against_eval(pool, ev_hashes)
        sub = Rl.ladder_subset(ded["kept"], RECIPE["n"])
        sub = Rl.shuffle_labels(sub, seed=a.seed)
        examples = [Rl.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
        arep = Rl.attach_adapters(m_after, rank=RECIPE["rank"], seed=a.seed, alpha=RECIPE["alpha"])
        m_after.to(device)
        print(f"[HELM] successor adapters: {arep['n_modules']} modules, "
              f"{arep['trainable_params']:,} trainable", flush=True)
        train_rep = Rl.train_adapter_qa(m_after, tok, device, examples, lr=RECIPE["lr"],
                                        epochs=RECIPE["epochs"], batch_size=RECIPE["batch"],
                                        warmup_ratio=RECIPE["warmup"], seed=a.seed, micro_batch=2)
        m_after.eval()
        rep["before_repo"] = "gemma-2-2b:frozen-base"
        rep["after_repo"] = "gemma-2-2b:+shuffled-label-LoRA"
        rep["successor_recipe"] = RECIPE
        rep["successor_train"] = {k: train_rep.get(k) for k in
                                  ("steps", "final_loss", "mean_loss_last_epoch", "tokens_seen")}
        rep["successor_adapters"] = {k: arep[k] for k in ("n_modules", "trainable_params", "rank", "alpha")}
        print(f"[HELM] successor train: steps={train_rep.get('steps')} "
              f"final_loss={train_rep.get('final_loss')}", flush=True)
    else:
        print(f"[HELM] loading before={a.before_repo} after={a.after_repo}", flush=True)
        m_before = AutoModelForCausalLM.from_pretrained(
            a.before_repo, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager").eval()
        m_after = AutoModelForCausalLM.from_pretrained(
            a.after_repo, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager").eval()
        tok = AutoTokenizer.from_pretrained(a.after_repo)

    # ---- (1) paired residual harvest + CKA ----
    texts = _harvest_texts(a.n_per_subject)
    print(f"[HELM] harvest corpus: {len(texts)} MMLU items", flush=True)
    acts_before, acts_after = harvest_paired_text(m_before, m_after, a.layer, texts, tok, device)
    n_tok, d = acts_before.shape
    cka = HC.linear_cka(acts_before.numpy(), acts_after.numpy())
    rep["n_paired_tokens"] = int(n_tok)
    rep["d_model"] = int(d)
    rep["cka"] = cka
    # CKA positive control: plant a direction into the BEFORE side; CKA(before, planted) MUST drop
    # below ~0.9 (proves CKA is sensitive), so a high CKA on the real pair means genuinely-preserved,
    # not a coarse instrument. Replaces the crosscoder-plant control (void per the 2026-09-01 smoke).
    planted_b, _ub, plant_info = DM.plant_direction(acts_before, seed=a.seed)
    cka_planted = HC.linear_cka(acts_before.numpy(), planted_b.numpy())
    rep["cka_planted"] = cka_planted
    rep["cka_positive_control_fires"] = HC.cka_positive_control_ok(cka_planted)
    rep["plant_info"] = plant_info
    print(f"[HELM][1] n_paired_tokens={n_tok} d={d} CKA={cka:.4f} "
          f"CKA_planted={cka_planted:.4f} (control_fires={rep['cka_positive_control_fires']})",
          flush=True)
    _w()

    # ---- (2) crosscoder on the REAL pair -> {survival, lost, emergent} + FVU ----
    ae, cfg = XC.train_crosscoder(acts_before, acts_after, a.layer, a.seed, device,
                                  steps_override=steps)
    fr = XC.pair_fractions(ae, acts_before, acts_after, device)
    fv = XC.fvu(ae, acts_before, acts_after, device)
    real_verdict = HC.preservation_verdict(fr["survival_fraction"], fr["lost_fraction"],
                                           fr["emergent_fraction"], IDENTITY_BAND)
    rep["crosscoder_real"] = {"survival": fr["survival_fraction"], "lost": fr["lost_fraction"],
                              "emergent": fr["emergent_fraction"], "n_live": fr["n_live"],
                              "fvu_before": fv["fvu_before"], "fvu_after": fv["fvu_after"],
                              "verdict": real_verdict["verdict"], "cfg": cfg}
    print(f"[HELM][2] REAL survival={fr['survival_fraction']:.4f} lost={fr['lost_fraction']:.4f} "
          f"emergent={fr['emergent_fraction']:.4f} fvu_after={fv['fvu_after']:.3f} "
          f"-> {real_verdict['verdict']}", flush=True)
    del ae
    torch.cuda.empty_cache()
    _w()

    # ---- (3) positive control: plant a known direction into the AFTER side; the crosscoder MUST
    #          register it as MOVED. before-side = clean after acts, after-side = planted acts, so
    #          the ONLY difference between the two sides is the plant (detection is attributable to
    #          it, not confounded with the real before->after change). ----
    planted, u, plant_info = DM.plant_direction(acts_after, seed=a.seed)
    ae_p, cfg_p = XC.train_crosscoder(acts_after, planted, a.layer, a.seed, device,
                                      steps_override=steps)
    fr_p = XC.pair_fractions(ae_p, acts_after, planted, device)
    pc_fires = HC.positive_control_ok(fr_p["lost_fraction"], fr_p["emergent_fraction"], IDENTITY_BAND)
    planted_verdict = HC.preservation_verdict(fr_p["survival_fraction"], fr_p["lost_fraction"],
                                              fr_p["emergent_fraction"], IDENTITY_BAND)
    rep["crosscoder_planted"] = {"survival": fr_p["survival_fraction"], "lost": fr_p["lost_fraction"],
                                 "emergent": fr_p["emergent_fraction"], "n_live": fr_p["n_live"],
                                 "verdict": planted_verdict["verdict"],
                                 "positive_control_fires": bool(pc_fires), "plant": plant_info}
    print(f"[HELM][3] PLANTED survival={fr_p['survival_fraction']:.4f} lost={fr_p['lost_fraction']:.4f} "
          f"emergent={fr_p['emergent_fraction']:.4f} -> {planted_verdict['verdict']} "
          f"fires={pc_fires}", flush=True)
    del ae_p
    torch.cuda.empty_cache()
    _w()

    # ---- (4) behavioural readouts: MCQ + cloze. --real scores BOTH sides (to get the drop); the
    #          identity smoke scores the AFTER side only (before==after). Text-only (extracted LMs
    #          are plain causal LMs), so processor=None and mmlu_accuracy is correct. ----
    # In --real the AFTER/BEFORE are PaliGemma-2 VLMs (the extracted LM submodule is the trunk with
    # no lm_head -> no .logits). Score MMLU on the FULL VLMs via mmlu_blankimg (blank 224 image), the
    # REPRICE-consistent way to read PG2's deleted MMLU; eager attn is already set by build_before_after.
    if a.real:
        mcq = TE.mmlu_blankimg(pg_after, proc, device, n_per_subject=a.n_per_subject)
        clz = TE.mmlu_cloze(pg_after, tok, device, processor=proc, n_per_subject=a.n_per_subject)
    else:
        mcq = TE.mmlu_accuracy(m_after, tok, device, n_per_subject=a.n_per_subject)
        clz = TE.mmlu_cloze(m_after, tok, device, processor=None, n_per_subject=a.n_per_subject)
    rep["mmlu_accuracy"] = {"acc": mcq["acc"], "acc_pct": mcq["acc_pct"], "n": mcq["n"],
                            "n_subjects": mcq["n_subjects"], "n_shots": mcq.get("n_shots", 5)}
    rep["mmlu_cloze"] = {"acc": clz["acc"], "acc_pct": clz["acc_pct"], "n": clz["n"],
                         "n_subjects": clz["n_subjects"], "forward": clz["forward"]}
    print(f"[HELM][4] AFTER mmlu_acc={mcq['acc_pct']:.2f}% mmlu_cloze={clz['acc_pct']:.2f}% "
          f"(n={mcq['n']})", flush=True)
    _w()

    if a.real or a.successor:
        if a.real:
            mcq_b = TE.mmlu_blankimg(pg_before, proc, device, n_per_subject=a.n_per_subject)
            clz_b = TE.mmlu_cloze(pg_before, tok, device, processor=proc, n_per_subject=a.n_per_subject)
        else:
            mcq_b = TE.mmlu_accuracy(m_before, tok, device, n_per_subject=a.n_per_subject)
            clz_b = TE.mmlu_cloze(m_before, tok, device, processor=None, n_per_subject=a.n_per_subject)
        rep["mmlu_accuracy_before"] = {"acc": mcq_b["acc"], "acc_pct": mcq_b["acc_pct"], "n": mcq_b["n"]}
        rep["mmlu_cloze_before"] = {"acc": clz_b["acc"], "acc_pct": clz_b["acc_pct"], "n": clz_b["n"]}
        # chance MCQ = 0.25 (4 options). Drop = fraction of the ABOVE-CHANCE margin that was lost.
        marg_b = max(mcq_b["acc"] - 0.25, 1e-6)
        behavior_drop_mcq = float(max(0.0, (mcq_b["acc"] - mcq["acc"]) / marg_b))
        cmarg_b = max(clz_b["acc"] - 0.25, 1e-6)
        behavior_drop_cloze = float(max(0.0, (clz_b["acc"] - clz["acc"]) / cmarg_b))
        print(f"[HELM][4b] BEFORE mmlu_acc={mcq_b['acc_pct']:.2f}% cloze={clz_b['acc_pct']:.2f}% "
              f"-> mcq_drop={behavior_drop_mcq:.3f} cloze_drop={behavior_drop_cloze:.3f}", flush=True)
        _w()
    else:
        behavior_drop_mcq = 0.0 if a.before_repo == a.after_repo else float("nan")
        behavior_drop_cloze = 0.0 if a.before_repo == a.after_repo else float("nan")

    del m_before
    torch.cuda.empty_cache()

    # ---- fold into the frozen (CKA-primary) adjudication ----
    rep["adjudication"] = HC.adjudicate(
        cka_real=cka, cka_planted=cka_planted,
        behavior_drop_mcq=behavior_drop_mcq, behavior_drop_cloze=behavior_drop_cloze,
        cross_real={"survival": fr["survival_fraction"], "lost": fr["lost_fraction"],
                    "emergent": fr["emergent_fraction"]},
        identity_band=IDENTITY_BAND, fvu=fv["fvu_after"])
    _w()

    # ---- wiring verdict: CKA~1 (identity smoke) or CKA-plant control fires (real); mmlu numeric ----
    control_ok = HC.cka_positive_control_ok(cka_planted)
    wiring_valid = bool(
        (cka > 0.99 if not (a.real or a.successor) else True)
        and control_ok
        and (mcq["acc_pct"] == mcq["acc_pct"])  # not NaN
        and mcq["n"] > 0 and clz["n"] > 0)
    rep["wiring_valid"] = wiring_valid
    rep["elapsed_s"] = time.time() - t0
    _w()
    print("HELMHOLTZ_CONFIRM_OK " + json.dumps(
        {"cka": cka, "cka_planted": cka_planted, "verdict": rep["adjudication"]["verdict"],
         "mmlu_acc_pct": mcq["acc_pct"], "mmlu_cloze_pct": clz["acc_pct"],
         "wiring_valid": wiring_valid}, default=str), flush=True)
    if a.smoke:
        assert wiring_valid, f"SMOKE wiring INVALID — see {out / 'metrics.json'}"


if __name__ == "__main__":
    main()
