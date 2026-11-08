"""XGEN V1 — decoder-isolated deletion (swap-back shell, GPU). Prereg §9.2 + §9.5.

Run as a standalone Gemma-1 text model, does the PaliGemma-1 LM decoder DELETE free-standing
knowledge capability vs base Gemma-1, isolated to the decoder weights (W11-A method, ported
cross-generation)? The §6 V1 verdict needs ALL THREE conjuncts — this one job emits all three:

  S_text = (acc_AFTER − acc_FLOOR)/(acc_BEFORE − acc_FLOOR), read on the sha1 discovery/confirmation
  halves (headline = confirmation), AND
  OOD diagnostic: mmlu_blankimg on the RELEASED PaliGemma-1 (native in-distribution forward) —
  if AFTER recovers there, the swap-back loss was a forward artifact, not genuine, AND
  RECOVERY GATE (W14-R, REUSED): does any frozen prompt transform recover AFTER's capability? If the
  best transform stays at floor (verdict H_deletion), the loss is weight-level deletion, not
  inference-time suppression. REUSES the frozen battery infra.pg.recovery_transforms (13 no-i18n
  transforms: identity + 7 format + 2 reframe + 3 few-shot) + scorer infra.pg.recovery_eval, the
  max-statistic bootstrap + the §6 verdict, τ_rec=0.5, δ_rec from BEFORE. (max_t R_recover answers
  "any transform recovers"; w14_arm3's min-over-transforms answers a different, restore-robustness
  question.) Split by content_hash = per_item_hash so it shares the S_text item space.

  V1 descriptive label: H_forget iff S_text upper-CI < θ_forget AND not OOD-recovers AND recovery
  verdict == H_deletion. Binding verdict applied at unblind §9.6 under BH-FDR.

Arms (swap-back shell, identical forward path): standalone base (positive control), BEFORE =
swap-back(base decoder) = S_text denom, AFTER = swap-back(PG1 decoder), FLOOR = swap-back(random).
Validity gates (HALT): positive control sane; extraction_ok; floor_ok; denom_ok. Fail-loud.

    pg run infra.aws_pg.jobs.xgen_deletion --type p5.4xlarge,g6e.8xlarge \
        --s3-prefix results/xgen_deletion --budget-min 220
"""
from __future__ import annotations

import argparse
import json
import pathlib

# §4 frozen held-out subjects (W14-R, REUSED verbatim): the recovery generalization set.
HELD_OUT = ["college_chemistry", "astronomy", "machine_learning", "college_mathematics",
            "philosophy", "high_school_world_history", "professional_law", "econometrics",
            "us_foreign_policy", "nutrition"]


def _recovery_gate(before_model, after_model, tok, device, floor_acc, *,
                   n_score, n_score_held, n_shots, frac, n_boot, seed, base_drop_margin):
    """§6 V1 conjunct-3 — 'no prompt transform recovers' via the FROZEN W14-R recovery gate.

    REUSE: recovery_transforms (13 no-i18n transforms, max_stat_bootstrap, delta_rec, verdict) +
    recovery_eval (load_mmlu, score_prompts, text-only). δ_rec + τ_rec depend only on BEFORE/FLOOR
    (controls) → AFTER-blind by construction. Split discovery/confirmation by content_hash (=
    per_item_hash). Returns a JSON-able dict; no_prompt_transform_recovers = (verdict==H_deletion).
    """
    import numpy as np
    from infra.pg import recovery_transforms as RT
    from infra.pg import recovery_eval as RE

    transforms = [t for t in RT.transform_set() if t[2] not in RT.CONJ_LANG]   # 13 confound-free
    assert len(transforms) == 13, f"expected 13 no-i18n transforms, got {len(transforms)}"
    ident = next(t for t in transforms if t[0] == "identity")
    dev_items, held_items, exemplars = RE.load_mmlu(HELD_OUT, n_per_subject=None, langs=(), n_shots=n_shots)

    rng = np.random.default_rng(seed)
    def _sub(items, n):
        if n < 0 or n >= len(items):
            return items
        return [items[i] for i in sorted(rng.choice(len(items), n, replace=False))]

    def _score(model, items, t):
        n_sh = RT.SHOTS[t[3]][0]
        prompts = [RT.apply_transform(it, t, exemplars.get(it["subject"], [])[:n_sh]) for it in items]
        gold = [it["answer"] for it in items]
        return RE.score_prompts(model, tok, device, prompts, RT.answer_candidates(t), gold)

    # --- δ_rec + disqualification: BEFORE under every transform on the DISCOVERY half (control) ---
    dev = _sub(dev_items, n_score)
    disc = [it for it in dev if int(RT.content_hash(it), 16) % 2 == 0]
    conf = [it for it in dev if int(RT.content_hash(it), 16) % 2 == 1]
    base_acc = {t[0]: float(np.mean(_score(before_model, disc, t))) for t in transforms}
    before_id = base_acc["identity"]
    # denom guard (mirrors w14r preflight denom_ok): a healthy base−floor headroom is required for
    # R_recover. Fail loud with a clear message instead of the cryptic per-resample assert inside
    # max_stat_bootstrap (which fires on an unlucky tiny-n resample when base≈floor).
    assert before_id - floor_acc > 0.05, (
        f"recovery gate degenerate: BEFORE identity acc {before_id:.4f} − floor {floor_acc:.4f} "
        f"≤ 0.05 on the discovery half (n={len(disc)}); raise --rec-n-score or check the instrument.")
    disq = [tid for tid, a in base_acc.items() if a < before_id - base_drop_margin]
    kept = {tid: a for tid, a in base_acc.items() if tid not in disq}
    delta = RT.delta_rec(kept, before_id, floor_acc)
    tau = RT.tau_rec(RT.r_recover(before_id, before_id, floor_acc), frac)      # swap-restore R = 1 → τ=frac

    # --- confirmation + held-out: AFTER under every transform, paired max-statistic vs BEFORE/floor ---
    def _boot(items):
        base_correct = _score(before_model, items, ident)
        cbt = {t[0]: _score(after_model, items, t) for t in transforms}
        return RT.max_stat_bootstrap(cbt, base_correct, floor_acc, n_boot=n_boot, seed=seed)

    cboot = _boot(conf)
    hboot = _boot(_sub(held_items, n_score_held))
    verdict = RT.verdict(cboot["ci_lo"], cboot["ci_hi"], hboot["ci_lo"],
                         swap_restore_r=1.0, base_spread=delta, frac=frac)
    return {
        "n_transforms": len(transforms), "transform_ids": [t[0] for t in transforms],
        "n_disc": len(disc), "n_conf": len(conf), "n_held": min(n_score_held, len(held_items)),
        "delta_rec": delta, "tau_rec": tau, "disqualified": disq,
        "confirmation": cboot, "held_out": hboot, "verdict": verdict,
        "no_prompt_transform_recovers": bool(verdict == "H_deletion"),
        "note": "REUSE W14-R battery+gate (max_t R_recover); split by content_hash=per_item_hash",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-revision", default="main")
    ap.add_argument("--floor-seed", type=int, default=1337)
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0, help="0 = full MMLU test split")
    ap.add_argument("--ppl-max-tokens", type=int, default=120_000)
    ap.add_argument("--published-mmlu", type=float, default=None,
                    help="recomputed Gemma-1 MMLU anchor (cite-verified at §9.0); None → sanity gate only")
    ap.add_argument("--tol-mmlu", type=float, default=3.0, help="abs tol (pct pts) vs --published-mmlu")
    ap.add_argument("--sesoi", type=float, default=0.10, help="power gate Δ_min on (base−floor) acc")
    # recovery-gate (W14-R reuse) params:
    ap.add_argument("--rec-n-score", type=int, default=1500, help="size-weighted dev subsample for recovery")
    ap.add_argument("--rec-n-score-held", type=int, default=600, help="held-out subsample for recovery")
    ap.add_argument("--rec-n-shots", type=int, default=5)
    ap.add_argument("--rec-frac", type=float, default=0.5, help="τ_rec = frac × R_swap-restore (frozen 0.5)")
    ap.add_argument("--rec-n-boot", type=int, default=2000)
    ap.add_argument("--rec-base-drop-margin", type=float, default=0.10)
    ap.add_argument("--no-recovery", action="store_true", help="skip the recovery gate (S_text+OOD only)")
    ap.add_argument("--smoke", action="store_true",
                    help="§9.2 wiring smoke: tiny MMLU (n_per_subject=2) + short ppl + tiny recovery")
    args = ap.parse_args()
    if args.smoke:
        args.mmlu_n_per_subject = 2
        args.ppl_max_tokens = 4096
        args.rec_n_score = 300        # ≥~150/half so the max-stat bootstrap stays non-degenerate
        args.rec_n_score_held = 150
        args.rec_n_boot = 200

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                                   # FIRST — gated metadata 401s otherwise

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PaliGemmaProcessor, \
        PaliGemmaForConditionalGeneration

    from infra.pg import swap_back as SB
    from infra.pg import text_eval as TE
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "XGEN V1 deletion needs GPU."
    device = "cuda"
    n_per = args.mmlu_n_per_subject or None
    base_repo = ENTITIES["L0_g1"].repo_id              # google/gemma-2b (Gemma-1)
    pg1_repo = ENTITIES["PG1"].repo_id                 # google/paligemma-3b-pt-224
    tok = AutoTokenizer.from_pretrained(base_repo, revision=args.base_revision)
    print(f"[XGEN-V1] base={base_repo}@{args.base_revision} pg1={pg1_repo} device={device}")

    # --- extract base + PG1 decoder state_dicts ONCE (bf16, matches the eval models) ---
    _b, base_dec, pg1_dec, meta = SB.load_xgen_decoders(
        device="cpu", base_revision=args.base_revision, dtype=torch.bfloat16)
    del _b
    torch.cuda.empty_cache()

    def _fresh_base():
        return AutoModelForCausalLM.from_pretrained(
            base_repo, revision=args.base_revision, torch_dtype=torch.bfloat16,
            device_map=device, attn_implementation="sdpa").eval()

    def _eval(model, label):
        m = TE.mmlu_accuracy(model, tok, device, n_per_subject=n_per)
        p = TE.clean_text_perplexity(model, tok, device, max_tokens=args.ppl_max_tokens)
        print(f"[XGEN-V1] {label:8s} MMLU={m['acc_pct']:.2f} ppl={p['ppl']:.2f}")
        return m, p

    # --- positive control: standalone base gemma-2b (harness validity, the Gemma-1 anchor) ---
    base_model = _fresh_base()
    base_mmlu, base_ppl = _eval(base_model, "base*")
    del base_model
    torch.cuda.empty_cache()

    # --- BEFORE (kept alive for the recovery gate) = swap-back(base decoder), identity swap ---
    before_model, before_info = SB.assemble_swapback_model(_fresh_base(), base_dec, device)
    before_mmlu, before_ppl = _eval(before_model, "BEFORE")

    # --- AFTER (kept alive for the recovery gate) = swap-back(PG1 decoder), the treatment ---
    after_model, after_info = SB.assemble_swapback_model(_fresh_base(), pg1_dec, device)
    after_mmlu, after_ppl = _eval(after_model, "AFTER")

    # --- FLOOR = swap-back(random decoder); only its identity acc is needed → eval then free ---
    floor_sd = SB.random_decoder_state_dict(base_dec, seed=args.floor_seed)
    floor_model, _ = SB.assemble_swapback_model(_fresh_base(), floor_sd, device)
    floor_mmlu, floor_ppl = _eval(floor_model, "FLOOR")
    del floor_model
    torch.cuda.empty_cache()

    # --- item-order alignment across the three arms (paired bootstrap correctness) ---
    item_hashes = before_mmlu["per_item_hash"]
    assert after_mmlu["per_item_hash"] == item_hashes == floor_mmlu["per_item_hash"], \
        "MMLU item order differs across BEFORE/AFTER/FLOOR — paired bootstrap would misalign."

    # --- S_text on discovery / confirmation / all halves (headline = confirmation) ---
    s_blocks = {}
    for which in ("confirmation", "discovery", "all"):
        a = SB.subset_by_hash(after_mmlu["per_item"], item_hashes, which)
        b = SB.subset_by_hash(before_mmlu["per_item"], item_hashes, which)
        f = SB.subset_by_hash(floor_mmlu["per_item"], item_hashes, which)
        point = SB.s_text(a, b, f)
        mean, lo, hi, nb = SB.boot_s_text(a, b, f)
        s_blocks[which] = {"S_text": point, "boot_mean": mean, "ci95": [lo, hi], "n": len(a),
                           "n_boot_valid": nb}

    # --- θ_forget = FLOOR + eval-noise band (S_text units), CALIBRATED ON THE DISCOVERY HALF ---
    before_disc = SB.subset_by_hash(before_mmlu["per_item"], item_hashes, "discovery")
    floor_disc = SB.subset_by_hash(floor_mmlu["per_item"], item_hashes, "discovery")
    theta = SB.theta_forget(before_disc, floor_disc)

    # --- RECOVERY GATE (W14-R reuse) — needs BEFORE + AFTER alive; then free them ---
    if args.no_recovery:
        recovery = {"skipped": True, "no_prompt_transform_recovers": None, "verdict": None}
    else:
        recovery = _recovery_gate(
            before_model, after_model, tok, device, floor_mmlu["acc"],
            n_score=args.rec_n_score, n_score_held=args.rec_n_score_held, n_shots=args.rec_n_shots,
            frac=args.rec_frac, n_boot=args.rec_n_boot, seed=args.floor_seed,
            base_drop_margin=args.rec_base_drop_margin)
        print(f"[XGEN-V1] recovery gate: max_R={recovery['confirmation']['max_r']:.3f} "
              f"CI=[{recovery['confirmation']['ci_lo']:.3f},{recovery['confirmation']['ci_hi']:.3f}] "
              f"τ={recovery['tau_rec']:.3f} δ={recovery['delta_rec']:.3f} → {recovery['verdict']}")
    del before_model, after_model
    torch.cuda.empty_cache()

    # --- OOD diagnostic: released PaliGemma-1, blank-image in-distribution forward ---
    pg1 = PaliGemmaForConditionalGeneration.from_pretrained(
        pg1_repo, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa").eval()
    proc = PaliGemmaProcessor.from_pretrained(pg1_repo)
    ood = TE.mmlu_blankimg(pg1, proc, device, n_per_subject=n_per)
    assert ood["per_item_hash"] == item_hashes, \
        "OOD blank-image item order differs from the swap-back arms — per-item OOD pairing would misalign."
    print(f"[XGEN-V1] OOD(blank-img, native PG1) MMLU={ood['acc_pct']:.2f}")
    del pg1
    torch.cuda.empty_cache()

    # OOD recovery: native PG1 does MMLU well above BOTH floor AND the swap-back AFTER → the swap-back
    # loss was a forward/embed artifact, not genuine decoder-level deletion (§5 OOD control, W14-R).
    ood_recovers = bool(ood["acc_pct"] > floor_mmlu["acc_pct"] + 5.0
                        and ood["acc_pct"] > after_mmlu["acc_pct"] + 5.0)

    # --- descriptive V1 label: all three §6 conjuncts (binding verdict at unblind §9.6, BH-FDR) ---
    conf_block = s_blocks["confirmation"]
    no_transform_recovers = recovery.get("no_prompt_transform_recovers")
    descriptive_v1 = ("H_forget" if (conf_block["ci95"][1] < theta and not ood_recovers
                                     and no_transform_recovers is not False) else "H_retain")

    # --- validity gates (mirror W11-A controls; HALT-worthy, reported) ---
    extraction_delta = abs(before_mmlu["acc_pct"] - base_mmlu["acc_pct"])
    extraction_ok = bool(extraction_delta <= 2.0)
    floor_ok = bool(floor_mmlu["acc_pct"] <= 35.0)          # 4-way chance = 25%
    denom = before_mmlu["acc"] - floor_mmlu["acc"]
    denom_ok = bool(denom > 0.05)
    headroom = base_mmlu["acc"] - floor_mmlu["acc"]
    power_ok = bool(headroom >= args.sesoi)
    if args.published_mmlu is not None:
        pc_delta = abs(base_mmlu["acc_pct"] - args.published_mmlu)
        positive_ok = bool(pc_delta <= args.tol_mmlu)
    else:
        pc_delta = None
        positive_ok = bool(base_mmlu["acc_pct"] >= 30.0)    # sanity floor only; set anchor at §9.0
    ppl_ratio = after_ppl["ppl"] / base_ppl["ppl"]
    controls_ok = bool(positive_ok and extraction_ok and floor_ok and denom_ok)

    result = {
        "torch": torch.__version__, "meta": meta,
        "swap_back": {"before": before_info, "after": after_info},
        "mmlu": {
            "base":   {"acc_pct": base_mmlu["acc_pct"], "acc": base_mmlu["acc"], "n": base_mmlu["n"]},
            "BEFORE": {"acc_pct": before_mmlu["acc_pct"], "acc": before_mmlu["acc"],
                       "per_item": before_mmlu["per_item"]},
            "AFTER":  {"acc_pct": after_mmlu["acc_pct"], "acc": after_mmlu["acc"],
                       "per_item": after_mmlu["per_item"]},
            "FLOOR":  {"acc_pct": floor_mmlu["acc_pct"], "acc": floor_mmlu["acc"],
                       "per_item": floor_mmlu["per_item"]},
            "item_hashes": item_hashes, "n_per_subject": args.mmlu_n_per_subject,
        },
        "OOD_blankimg_native_pg1": {"acc_pct": ood["acc_pct"], "acc": ood["acc"],
                                    "per_item": ood["per_item"], "per_item_hash": ood["per_item_hash"],
                                    "recovers": ood_recovers},
        "S_text": s_blocks,
        "theta_forget": theta,
        "recovery_gate": recovery,
        "descriptive_v1_label": descriptive_v1,
        "descriptive_v1_note": ("DESCRIPTIVE: confirmation S_text upper-CI < θ_forget AND not OOD-recovers "
                                "AND recovery verdict==H_deletion; binding verdict at unblind §9.6 (BH-FDR)"),
        "perplexity": {"ppl_after": after_ppl["ppl"], "ppl_base": base_ppl["ppl"],
                       "ppl_before": before_ppl["ppl"], "ppl_floor": floor_ppl["ppl"],
                       "ratio_after_over_base": ppl_ratio,
                       "after_per_window": after_ppl["per_window"],
                       "base_per_window": base_ppl["per_window"]},
        "power_gate": {"headroom_base_minus_floor": headroom, "sesoi": args.sesoi, "power_ok": power_ok,
                       "primary_axis": "MMLU" if power_ok else "perplexity_ratio"},
        "controls": {
            "positive_anchor": args.published_mmlu, "positive_delta": pc_delta, "positive_ok": positive_ok,
            "extraction_delta_pp": extraction_delta, "extraction_ok": extraction_ok,
            "floor_mmlu_pct": floor_mmlu["acc_pct"], "floor_ok": floor_ok,
            "denom": denom, "denom_ok": denom_ok, "CONTROLS_OK": controls_ok},
        "firewall": "labels-only vs PG2 (W11-A 'H_forget'); PG1 thresholds on PG1 control data",
        "label_discipline": "CONFIRMATORY (decoder-isolation swap-back = an intervention)",
    }
    (out / "deletion.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[XGEN-V1] S_text(confirmation)={conf_block['S_text']:.4f} CI={conf_block['ci95']} "
          f"θ_forget={theta:.4f} | recovery={recovery.get('verdict')} | OOD_recovers={ood_recovers} "
          f"→ {descriptive_v1} (descriptive) | CONTROLS_OK={controls_ok} | power_ok={power_ok}")
    if not controls_ok:
        raise RuntimeError(
            f"XGEN V1 CONTROLS HALT: positive_ok={positive_ok} extraction_ok={extraction_ok} "
            f"(|Δ|={extraction_delta:.2f}) floor_ok={floor_ok} denom_ok={denom_ok} (denom={denom:.4f}). "
            f"See {out / 'deletion.json'}.")


if __name__ == "__main__":
    main()
