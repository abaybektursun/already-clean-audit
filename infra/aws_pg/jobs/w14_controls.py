"""W14 §9 STEP-2 — re-freeze the forgetting-axis controls (denominators) for THIS study.

Writes a FRESH legA_controls.json that Arm 2 (R_mmlu) and Arm 3 (R_restore) read — NOT W11's
verbatim file (the pre/post-PR#20 best-checkpoint boundary makes the old textual numbers
inadmissible; §0 baseline discipline). Clones the w11_legA_controls pattern, but the BINDING
forward condition is **blank-image** (gray 224², PaliGemma in-distribution), per prereg §1.

  base   = standalone gemma-2-2b, TEXT-ONLY            (published 51.3 validity anchor + provenance)
  BEFORE = decoder-reset PG2 (build_before_after)      → the R_mmlu denominator base
  FLOOR  = random-decoder PG2 (build_random_after), ≥3 seeds → the R_mmlu 0-anchor (binding = seed 1337)
  AFTER  = the real PG2-LM: SEALED (built + deleted; never scored here — the project MDE, §9-5).

Primary scale (binding): BLANK-IMAGE BEFORE/FLOOR (mmlu_blankimg). Text-only BEFORE/FLOOR logged
as the secondary sensitivity regime. Validity gates (controls-beat-null, never zero):
  * floor_ok      : FLOOR blank-image MMLU ≤ 35% (4-way chance) → the 0-anchor is dead.
  * denom_ok      : BEFORE_blank − FLOOR_blank > 0.05 → an informative scale.
  * sesoi_grounded: SESOI ≥ 2·CI-halfwidth on BEFORE_blank per-item → SESOI above sampling noise.
  NO extraction_ok (before≈base) gate: BEFORE is a base-decoder + PG-extended-embed CHIMERA on the
  blank-image forward; a |before−base| equality gate is unregistered and false-by-design (§A2.9).

Fail-loud; no try/except.

    python -m infra.aws_pg.jobs.w14_controls --out <dir> [--mmlu-n-per-subject 0] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib

FLOOR_SEEDS = (1337, 1338, 1339)        # ≥3 random-decoder seeds; binding = the first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pg-variant", choices=["pt", "mix"], default="pt",
                    help="base→pt (W14) or base→mix (second-ΔW generality gate)")
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0, help="0 = full MMLU test split")
    ap.add_argument("--ppl-max-tokens", type=int, default=120_000)
    ap.add_argument("--sesoi", type=float, default=0.10)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny/fast sizes to exercise the wiring (2 items/subject, short ppl)")
    args = ap.parse_args()
    if args.smoke:
        args.mmlu_n_per_subject = 2
        args.ppl_max_tokens = 4_000

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from infra.pg import diff_metrics as DM
    from infra.pg import text_eval as TE
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "W14 controls need GPU."
    device = "cuda"
    print(f"[W14-CTRL] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} smoke={args.smoke}")
    L0 = ENTITIES["L0"]
    gemma_tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    n_per = args.mmlu_n_per_subject or None

    # ---- base gemma-2-2b TEXT-ONLY (published anchor + provenance; eager for soft-capping) ----
    base = AutoModelForCausalLM.from_pretrained(
        L0.repo_id, revision=L0.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    base_mmlu = TE.mmlu_accuracy(base, gemma_tok, device, n_per_subject=n_per)
    base_ppl = TE.clean_text_perplexity(base, gemma_tok, device, max_tokens=args.ppl_max_tokens)
    print(f"[W14-CTRL] base text-only MMLU={base_mmlu['acc_pct']:.2f} ppl={base_ppl['ppl']:.2f}")
    del base
    torch.cuda.empty_cache()

    # ---- build AFTER + BEFORE; SEAL AFTER ----
    pg_key = "PG_mix" if args.pg_variant == "mix" else "PG"
    print(f"[W14-CTRL] pg_variant={args.pg_variant} (pg_key={pg_key})", flush=True)
    pg_after, pg_before, proc, paths, _it, gate = H.build_before_after(device, pg_key=pg_key)
    del pg_after                                   # AFTER is the sealed MDE treatment — never scored here
    torch.cuda.empty_cache()

    # ---- BEFORE: blank-image (primary) + text-only (secondary) + ppl ----
    before_blank = TE.mmlu_blankimg(pg_before, proc, device, n_per_subject=n_per, gray=128)
    before_text = TE.mmlu_accuracy(pg_before, gemma_tok, device, n_per_subject=n_per)
    before_ppl = TE.clean_text_perplexity(pg_before, gemma_tok, device, max_tokens=args.ppl_max_tokens)
    print(f"[W14-CTRL] BEFORE blank={before_blank['acc_pct']:.2f} text={before_text['acc_pct']:.2f} "
          f"ppl={before_ppl['ppl']:.2f}")

    # ---- FLOOR: ≥3 random-decoder seeds, blank-image; binding = FLOOR_SEEDS[0] (+ its text-only) ----
    floor_blank_seeds, floor_blank_binding, floor_text_binding, floor_ppl_binding = {}, None, None, None
    for si, seed in enumerate(FLOOR_SEEDS):
        pg_random = H.build_random_after(pg_before, paths, device, seed=seed)
        fb = TE.mmlu_blankimg(pg_random, proc, device, n_per_subject=n_per, gray=128)
        floor_blank_seeds[seed] = fb
        print(f"[W14-CTRL] FLOOR seed={seed} blank MMLU={fb['acc_pct']:.2f}")
        if si == 0:
            floor_blank_binding = fb
            floor_text_binding = TE.mmlu_accuracy(pg_random, gemma_tok, device, n_per_subject=n_per)
            floor_ppl_binding = TE.clean_text_perplexity(pg_random, gemma_tok, device,
                                                         max_tokens=args.ppl_max_tokens)
        del pg_random
        torch.cuda.empty_cache()
    del pg_before
    torch.cuda.empty_cache()

    # ---- item-order fingerprint: every MMLU eval must share the deterministic order ----
    item_hashes = before_blank["per_item_hash"]
    for name, d in (("base", base_mmlu), ("before_text", before_text),
                    ("floor_blank", floor_blank_binding), ("floor_text", floor_text_binding)):
        assert d["per_item_hash"] == item_hashes, f"MMLU item order differs in {name} — nondeterministic load."

    # ---- validity gates (blank-image is the binding scale; no extraction_ok) ----
    denom_blank = before_blank["acc"] - floor_blank_binding["acc"]
    denom_text = before_text["acc"] - floor_text_binding["acc"]
    floor_ok = bool(floor_blank_binding["acc_pct"] <= 35.0)
    denom_ok = bool(denom_blank > 0.05)
    # §A2.7: paired item-resample bootstrap, n_boot=2000, seed 1337 (project convention, not the
    # DM.bootstrap_ci n_boot=1000 default) — the SESOI grounding CI must use the frozen draw count.
    before_ci = DM.bootstrap_ci(np.array(before_blank["per_item"], dtype=float),
                                n_boot=2000, seed=1337, stat="mean")
    ci_halfwidth = (before_ci["hi"] - before_ci["lo"]) / 2.0
    sesoi_grounded = bool(args.sesoi >= 2.0 * ci_halfwidth)
    floor_seed_spread = float(np.std([floor_blank_seeds[s]["acc"] for s in FLOOR_SEEDS]))

    controls_ok = bool(floor_ok and denom_ok and sesoi_grounded)
    print(f"[W14-CTRL] FLOOR_blank={floor_blank_binding['acc_pct']:.2f}(ok={floor_ok}) "
          f"denom_blank={denom_blank:.4f}(ok={denom_ok}) "
          f"SESOI {args.sesoi} vs 2·ci_hw {2*ci_halfwidth:.4f}(grounded={sesoi_grounded}) "
          f"floor_seed_spread={floor_seed_spread:.4f} → controls_ok={controls_ok}")

    def _mmlu_block(d):
        return {"acc_pct": d["acc_pct"], "acc": d["acc"], "n": d["n"], "per_item": d["per_item"]}

    result = {
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "harness": "MMLU 5-shot single-token-answer; PRIMARY=blank-image (gray128), SECONDARY=text-only",
        "pg_variant": args.pg_variant,
        "forward_primary": "blank-image", "gray": 128,
        # the standalone base text-only published-check loads eager (Gemma2 soft-capping); the BINDING
        # blank-image/text BEFORE+FLOOR forwards go through H.build_before_after (the impl is set there
        # — same path the arms reuse, so controls↔arms are consistent regardless of the value).
        "attn_impl": {"base_textonly_published_check": "eager",
                      "binding_before_floor": "set by diff_harvest.build_before_after"},
        "mmlu_n_per_subject": args.mmlu_n_per_subject, "sesoi": args.sesoi, "smoke": args.smoke,
        "bootstrap": {"n_boot": 2000, "seed": 1337, "paired": True},
        "item_hashes": item_hashes,
        "build_before_after_gate": gate,
        "floor_seeds": list(FLOOR_SEEDS), "floor_seed_spread_acc": floor_seed_spread,
        # PRIMARY (binding) — blank-image
        "mmlu_blank": {
            "BEFORE": _mmlu_block(before_blank),
            "FLOOR": _mmlu_block(floor_blank_binding),
            "FLOOR_per_seed": {str(s): floor_blank_seeds[s]["acc_pct"] for s in FLOOR_SEEDS},
        },
        # SECONDARY — text-only (sensitivity)
        "mmlu_text": {
            "base": {**_mmlu_block(base_mmlu), "published": 51.3},
            "BEFORE": _mmlu_block(before_text),
            "FLOOR": _mmlu_block(floor_text_binding),
        },
        "perplexity": {
            "base": {"ppl": base_ppl["ppl"], "per_window": base_ppl["per_window"]},
            "BEFORE": {"ppl": before_ppl["ppl"], "per_window": before_ppl["per_window"]},
            "FLOOR": {"ppl": floor_ppl_binding["ppl"], "per_window": floor_ppl_binding["per_window"]},
        },
        # BINDING denominator = blank-image BEFORE − FLOOR (the R_mmlu / R_restore scale)
        "frozen_denominator_mmlu_acc": denom_blank,
        "frozen_denominator_mmlu_acc_text": denom_text,
        "checks": {
            "floor_blank_pct": floor_blank_binding["acc_pct"], "floor_ok": floor_ok,
            "denom_blank": denom_blank, "denom_ok": denom_ok,
            "before_blank_ci95": before_ci, "ci_halfwidth": ci_halfwidth,
            "sesoi_grounded": sesoi_grounded,
            "note": "no extraction_ok (before≈base) gate — BEFORE is a decoder/embed chimera on "
                    "the blank-image forward (§A2.9); MMLU greedy argmax is deterministic so the "
                    "noise floor is the per-item sampling CI above.",
        },
        "CONTROLS_OK": controls_ok,
    }
    (out / "legA_controls.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W14-CTRL] CONTROLS_OK={controls_ok}")
    if not controls_ok:
        raise RuntimeError(
            f"W14 CONTROLS HALT: floor_ok={floor_ok} (FLOOR={floor_blank_binding['acc_pct']:.2f}%) "
            f"denom_ok={denom_ok} (denom={denom_blank:.4f}) sesoi_grounded={sesoi_grounded}. "
            f"See {out / 'legA_controls.json'}.")
    print("W14_CONTROLS_OK", json.dumps({"denom_blank": round(denom_blank, 4),
                                         "floor_blank_pct": floor_blank_binding["acc_pct"]}))


if __name__ == "__main__":
    main()
