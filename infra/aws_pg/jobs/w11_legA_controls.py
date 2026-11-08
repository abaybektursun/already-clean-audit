"""W11 §9 Leg-A CONTROLS (step 2/4: develop-on-control, NO AFTER/PG2-LM peek).

The forgetting-axis scale anchors, all on the SAME text harness (MMLU 5-shot +
wikitext ppl), text-only (image tokens absent = §4 C2 parity):

  base   = standalone gemma-2-2b            (published positive control + validity anchor)
  BEFORE = decoder-reset PG2 (build_before_after: decoder+norm = base gemma-2-2b,
           embed/head/forward held identical to AFTER)  → the S_text denominator base
  FLOOR  = random-decoder PG2 (build_random_after)       → the S_text 0-anchor (N3)

Validity gates (controls-beat-null, never zero):
  * extraction_ok : BEFORE MMLU ≈ standalone base (|Δ|≤2 pts) → the PG embed/head do NOT
    confound the decoder contrast (the decoder-reset really equals base-gemma competence).
  * floor_ok      : random-decoder MMLU ≈ 4-way chance (≤35%) → the 0-anchor is dead.
  * denom_ok      : BEFORE−FLOOR > 0.05 → an informative scale (the CEILING−FLOOR analog).
  * struct probe  : a PaliGemma TEXT-ONLY forward yields finite .logits (the one unverified
    assumption of running PG2-LM as a text model) — fail-loud before any expensive eval.

AFTER (the real PG2-LM) is the SEALED treatment: built then immediately deleted here; it is
evaluated only at unblind (§9-5). Writes controls.json with per-item arrays (paired bootstrap
at unblind), the frozen denominator, and SESOI=0.10. Fail-loud; no try/except.

    python -m infra.aws_pg.jobs.w11_legA_controls --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0, help="0 = full MMLU test split")
    ap.add_argument("--ppl-max-tokens", type=int, default=120_000)
    ap.add_argument("--seed-floor", type=int, default=1337)
    ap.add_argument("--sesoi", type=float, default=0.10, help="retained-competence drop margin (§6A)")
    args = ap.parse_args()

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

    assert torch.cuda.is_available(), "Leg-A controls need GPU."
    device = "cuda"
    print(f"[W11-A-CTRL] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")

    L0 = ENTITIES["L0"]
    gemma_tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    n_per = args.mmlu_n_per_subject or None

    # --- build AFTER + BEFORE; SEAL AFTER (delete) after the text-only structural probe ---
    pg_after, pg_before, proc, paths, image_token_index, gate = H.build_before_after(device)
    probe_enc = gemma_tok(["The capital of France is"], return_tensors="pt").to(device)
    with torch.inference_mode():
        probe_out = pg_after(**probe_enc, logits_to_keep=1)
    assert hasattr(probe_out, "logits"), "PaliGemma text-only forward returned no .logits — Leg-A path invalid."
    assert torch.isfinite(probe_out.logits).all(), "non-finite logits on the text-only probe."
    probe_shape = list(probe_out.logits.shape)
    print(f"[W11-A-CTRL] AFTER text-only structural probe OK, logits shape={probe_shape}")
    del pg_after, probe_out
    torch.cuda.empty_cache()

    # probe BEFORE too (the model actually evaluated here) — validates its text-only dispatch
    # before any expensive MMLU loop (critic: probe was on the now-sealed AFTER).
    with torch.inference_mode():
        pb_out = pg_before(**probe_enc, logits_to_keep=1)
    assert hasattr(pb_out, "logits") and torch.isfinite(pb_out.logits).all(), \
        "BEFORE text-only forward invalid."
    del pb_out
    torch.cuda.empty_cache()

    # --- base gemma-2-2b (published positive control + validity anchor) ---
    # sdpa to MATCH BEFORE/FLOOR/AFTER (all PaliGemma sdpa) — removes the kernel confound on the
    # extraction_ok (BEFORE≈base) check (critic CRITICAL: eager-vs-sdpa parity).
    attn_impl = "sdpa"
    base = AutoModelForCausalLM.from_pretrained(
        L0.repo_id, revision=L0.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation=attn_impl).eval()
    base_mmlu = TE.mmlu_accuracy(base, gemma_tok, device, n_per_subject=n_per)
    base_ppl = TE.clean_text_perplexity(base, gemma_tok, device, max_tokens=args.ppl_max_tokens)
    print(f"[W11-A-CTRL] base   MMLU={base_mmlu['acc_pct']:.2f} ppl={base_ppl['ppl']:.2f}")
    del base
    torch.cuda.empty_cache()

    # --- BEFORE (decoder-reset PG2 = base-gemma decoder in the PG harness) ---
    before_mmlu = TE.mmlu_accuracy(pg_before, gemma_tok, device, n_per_subject=n_per)
    before_ppl = TE.clean_text_perplexity(pg_before, gemma_tok, device, max_tokens=args.ppl_max_tokens)
    print(f"[W11-A-CTRL] BEFORE MMLU={before_mmlu['acc_pct']:.2f} ppl={before_ppl['ppl']:.2f}")

    # --- FLOOR (random-decoder PG2) ---
    pg_random = H.build_random_after(pg_before, paths, device, seed=args.seed_floor)
    floor_mmlu = TE.mmlu_accuracy(pg_random, gemma_tok, device, n_per_subject=n_per)
    floor_ppl = TE.clean_text_perplexity(pg_random, gemma_tok, device, max_tokens=args.ppl_max_tokens)
    print(f"[W11-A-CTRL] FLOOR  MMLU={floor_mmlu['acc_pct']:.2f} ppl={floor_ppl['ppl']:.2f}")
    del pg_random, pg_before
    torch.cuda.empty_cache()

    # item-order fingerprint must be identical across the three arms (deterministic MMLU load)
    item_hashes = base_mmlu["per_item_hash"]
    assert before_mmlu["per_item_hash"] == item_hashes and floor_mmlu["per_item_hash"] == item_hashes, \
        "MMLU item order differs across base/BEFORE/FLOOR within controls — nondeterministic load."

    # --- validity gates ---
    before_vs_base = abs(before_mmlu["acc_pct"] - base_mmlu["acc_pct"])
    extraction_ok = bool(before_vs_base <= 2.0)
    floor_ok = bool(floor_mmlu["acc_pct"] <= 35.0)         # 4-way chance is 25%
    denom = before_mmlu["acc"] - floor_mmlu["acc"]
    denom_ok = bool(denom > 0.05)

    # N1 / eval-noise floor (§5A): MMLU is DETERMINISTIC (greedy argmax) → base-vs-base re-run gap
    # is exactly 0; the relevant noise floor is the per-item sampling CI. Ground SESOI in it.
    base_ci = DM.bootstrap_ci(np.array(base_mmlu["per_item"], dtype=float), stat="mean")
    ci_halfwidth = (base_ci["hi"] - base_ci["lo"]) / 2.0
    sesoi_grounded = bool(args.sesoi >= 2.0 * ci_halfwidth)   # SESOI well above sampling noise

    controls_ok = bool(extraction_ok and floor_ok and denom_ok and sesoi_grounded)
    print(f"[W11-A-CTRL] BEFORE−base |Δ|={before_vs_base:.2f} (extraction_ok={extraction_ok}) "
          f"| FLOOR chance-ok={floor_ok} | denom={denom:.4f} (ok={denom_ok}) "
          f"| SESOI {args.sesoi} vs 2·ci_hw {2*ci_halfwidth:.4f} (grounded={sesoi_grounded}) "
          f"→ controls_ok={controls_ok}")

    result = {
        "torch": torch.__version__,
        "harness": "MMLU 5-shot single-token-answer + wikitext-2-raw ppl, text-only",
        "attn_impl": attn_impl, "mmlu_n_per_subject": args.mmlu_n_per_subject,
        "item_hashes": item_hashes,
        "build_before_after_gate": gate,
        "text_only_probe_logits_shape": probe_shape,
        "sesoi": args.sesoi,
        "mmlu": {
            "base":   {"acc_pct": base_mmlu["acc_pct"], "acc": base_mmlu["acc"], "n": base_mmlu["n"],
                       "per_item": base_mmlu["per_item"], "published": 51.3, "ci95": base_ci},
            "BEFORE": {"acc_pct": before_mmlu["acc_pct"], "acc": before_mmlu["acc"], "n": before_mmlu["n"],
                       "per_item": before_mmlu["per_item"]},
            "FLOOR":  {"acc_pct": floor_mmlu["acc_pct"], "acc": floor_mmlu["acc"], "n": floor_mmlu["n"],
                       "per_item": floor_mmlu["per_item"]},
        },
        "perplexity": {
            "base":   {"ppl": base_ppl["ppl"], "per_window": base_ppl["per_window"]},
            "BEFORE": {"ppl": before_ppl["ppl"], "per_window": before_ppl["per_window"]},
            "FLOOR":  {"ppl": floor_ppl["ppl"], "per_window": floor_ppl["per_window"]},
        },
        "frozen_denominator_mmlu_acc": denom,         # BEFORE − FLOOR (the S_text scale, frozen)
        "checks": {
            "before_vs_base_abs_delta": before_vs_base, "extraction_ok": extraction_ok,
            "floor_mmlu_pct": floor_mmlu["acc_pct"], "floor_ok": floor_ok,
            "denom_ok": denom_ok,
            "eval_noise_floor_note": "MMLU greedy argmax is deterministic → base-vs-base gap = 0; "
                                     "sampling CI below is the noise floor",
            "base_acc_ci95": base_ci, "ci_halfwidth": ci_halfwidth, "sesoi_grounded": sesoi_grounded},
        "CONTROLS_OK": controls_ok,
    }
    (out / "legA_controls.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W11-A-CTRL] CONTROLS_OK={controls_ok}")
    if not controls_ok:
        raise RuntimeError(
            f"Leg-A CONTROLS HALT: extraction_ok={extraction_ok} (BEFORE−base |Δ|={before_vs_base:.2f}) "
            f"floor_ok={floor_ok} (FLOOR={floor_mmlu['acc_pct']:.2f}%) denom_ok={denom_ok} (denom={denom:.4f}) "
            f"sesoi_grounded={sesoi_grounded}. See {out / 'legA_controls.json'}.")
    print("[W11-A-CTRL] gate passed → scale frozen; AFTER (PG2-LM) evaluated only at unblind.")


if __name__ == "__main__":
    main()
