"""W14 §9 STEP-1 PREFLIGHT GATE (fail-fast, before the controls re-freeze or any arm).

Clones the w11_preflight pattern, scoped to W14's primary (blank-image) MMLU readout.
Gates (all on GPU; HALT/raise on any failure, no try/except):

  (1a) Generation firewall (§0): base gemma-2-2b config num_hidden_layers == 26 AND PG2's
       text_config num_hidden_layers == 26 (Gemma-2, NOT the Gemma-1/18L paligemma1 sibling).
  (1b) Harness validity: base gemma-2-2b reproduces its PUBLISHED 5-shot MMLU (51.3) within
       --tol-mmlu → the text instrument is sound. Plus tokenizer parity (gemma vs PG2 processor
       tokenizer encode shared-vocab text identically) and finite wikitext ppl.
  (1c) Blank-image instrument: build_pair("base-pt") then mmlu_blankimg on the AFTER model over
       a small per-subject cap — the PRIMARY readout must RUN; torch.isfinite asserted on the
       scored answer-surface logits (last-token + the four choice-letter logits, §A2.2(ii), HALT
       on non-finite); gray==128 asserted; per_item_hash matches the text-only per_item_hash.
  (1d) Item-split firewall: the blank-image per_item_hash splits by hash-parity into two
       non-empty, DISJOINT halves (the Arm-2/3 discovery/confirmation item-split mechanism).

This preflight gates HARNESS validity (vs the published number); the per-arm jobs separately
re-read the re-frozen legA_controls.json and assert base-MMLU/denom alignment (matches the W11
structure: w11_preflight = harness validity; w11_legA_unblind re-reads controls).

    python -m infra.aws_pg.jobs.w14_preflight --out <dir> [--mmlu-n-per-subject 0] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _hash_parity_split(hashes):
    """Deterministic item split by hash parity (int(h,16) % 2) — the Arm-2/3 item firewall."""
    disc = [h for h in hashes if int(h, 16) % 2 == 0]
    conf = [h for h in hashes if int(h, 16) % 2 == 1]
    return disc, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol-mmlu", type=float, default=3.0,
                    help="abs tol (pct pts) on base gemma-2-2b MMLU vs published 51.3")
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0,
                    help="cap test items/subject for the base validity MMLU (0 = full split)")
    ap.add_argument("--blankimg-n-per-subject", type=int, default=3,
                    help="small per-subject cap for the blank-image instrument finiteness probe")
    ap.add_argument("--ppl-max-tokens", type=int, default=120_000)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny/fast sizes to exercise the wiring (2 items/subject, short ppl)")
    args = ap.parse_args()
    if args.smoke:
        args.mmlu_n_per_subject = 2
        args.blankimg_n_per_subject = 2
        args.ppl_max_tokens = 4_000
        args.tol_mmlu = 100.0      # smoke exercises WIRING only; n=2/subject MMLU is far too noisy to
                                   # gate on the published number — the real (no-smoke) run uses tol 3.0

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()                                    # MUST precede transformers/datasets

    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PaliGemmaProcessor

    from infra.pg import text_eval as TE
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "W14 preflight needs GPU (model forwards)."
    device = "cuda"
    print(f"[W14-PRE] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} smoke={args.smoke}")

    api = HfApi()
    L0 = ENTITIES["L0"]            # google/gemma-2-2b
    PG = ENTITIES["PG"]            # google/paligemma2-3b-pt-224
    l0_sha = api.model_info(L0.repo_id, revision=L0.revision or "main").sha
    pg_sha = api.model_info(PG.repo_id).sha
    print(f"[W14-PRE] L0={L0.repo_id}@{l0_sha}  PG={PG.repo_id}@{pg_sha}")

    # ---- (1a) generation firewall: 26 decoder layers (Gemma-2, not the 18L sibling) ----
    l0_cfg = AutoConfig.from_pretrained(L0.repo_id, revision=L0.revision)
    pg_cfg = AutoConfig.from_pretrained(PG.repo_id)
    pg_text_cfg = getattr(pg_cfg, "text_config", pg_cfg)
    l0_layers = int(getattr(l0_cfg, "num_hidden_layers"))
    pg_layers = int(getattr(pg_text_cfg, "num_hidden_layers"))
    gate_1a = bool(l0_layers == 26 and pg_layers == 26)
    print(f"[W14-PRE][1a] firewall: L0 layers={l0_layers} PG text layers={pg_layers} → ok={gate_1a}")

    # ---- (1b) harness validity: base gemma-2-2b MMLU ≈ published + tokenizer parity + ppl ----
    model = AutoModelForCausalLM.from_pretrained(
        L0.repo_id, revision=L0.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()   # eager: Gemma2 soft-capping
    tok_g = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    tok_pg = PaliGemmaProcessor.from_pretrained(PG.repo_id).tokenizer
    parity = TE.tokenizer_parity(tok_g, tok_pg)
    n_per = args.mmlu_n_per_subject or None
    mmlu = TE.mmlu_accuracy(model, tok_g, device, n_per_subject=n_per)
    mmlu_delta = abs(mmlu["acc_pct"] - mmlu["published"])
    mmlu_ok = bool(mmlu_delta <= args.tol_mmlu)
    ppl = TE.clean_text_perplexity(model, tok_g, device, max_tokens=args.ppl_max_tokens)
    ppl_ok = bool(np.isfinite(ppl["ppl"]))
    gate_1b = bool(parity["match"] and mmlu_ok and ppl_ok)
    print(f"[W14-PRE][1b] parity={parity['match']} MMLU={mmlu['acc_pct']:.2f} vs {mmlu['published']} "
          f"|Δ|={mmlu_delta:.2f}(tol {args.tol_mmlu}) ppl={ppl['ppl']:.2f} → ok={gate_1b}")
    text_only_hashes = mmlu["per_item_hash"]
    del model
    torch.cuda.empty_cache()

    # ---- (1c) blank-image PRIMARY instrument runs + finite + gray==128 + item-order match ----
    pg_after, pg_before, proc, paths, _it, gate = H.build_before_after(device)
    bi = TE.mmlu_blankimg(pg_after, proc, device, n_per_subject=args.blankimg_n_per_subject, gray=128)
    # §A2.2(ii): non-finite logits → HALT. Assert torch.isfinite on the ACTUAL scored answer-surface
    # logits — the blank-image forward last-token logits + the four choice-letter logits the argmax
    # reads — not a vacuous acc∈[0,1] range check (acc is always in range by construction).
    from PIL import Image
    tok_proc = proc.tokenizer
    letter_ids = [tok_proc.encode(" " + L, add_special_tokens=False)[0] for L in TE.CHOICES]
    gray_img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    probe_prompt = TE._subject_header("abstract_algebra") + TE._q_block(
        "Which of the following is a group under addition?", ["Z", "Q+", "the empty set", "{1}"])
    probe = proc(text=[probe_prompt], images=[gray_img], return_tensors="pt", padding="longest").to(device)
    with torch.inference_mode():
        probe_last = pg_after(input_ids=probe["input_ids"], attention_mask=probe["attention_mask"],
                              pixel_values=probe["pixel_values"].to(torch.bfloat16),
                              logits_to_keep=1).logits[:, -1, :]
    logits_finite = bool(torch.isfinite(probe_last).all() and torch.isfinite(probe_last[:, letter_ids]).all())
    assert logits_finite, "W14 PREFLIGHT HALT: non-finite blank-image answer-surface logits (§A2.2)."
    bi_finite = bool(logits_finite and bi["n"] > 0)
    # the same per-subject cap on the text-only harness must produce the SAME item hashes/order
    after_text = TE.mmlu_accuracy(pg_after, tok_g, device, n_per_subject=args.blankimg_n_per_subject)
    order_match = bool(bi["per_item_hash"] == after_text["per_item_hash"])
    gray_ok = bool(bi["gray"] == 128)
    gate_1c = bool(bi_finite and order_match and gray_ok)
    print(f"[W14-PRE][1c] blank-image MMLU={bi['acc_pct']:.2f} (n={bi['n']}) logits_finite={logits_finite} "
          f"gray128={gray_ok} order_match_text={order_match} → ok={gate_1c}")
    del pg_after, pg_before
    torch.cuda.empty_cache()

    # ---- (1d) item-split firewall: hash-parity halves non-empty + disjoint ----
    disc, conf = _hash_parity_split(text_only_hashes)
    disjoint = bool(set(disc).isdisjoint(set(conf)))
    gate_1d = bool(len(disc) > 0 and len(conf) > 0 and disjoint)
    print(f"[W14-PRE][1d] item-split disc={len(disc)} conf={len(conf)} disjoint={disjoint} → ok={gate_1d}")

    ok = bool(gate_1a and gate_1b and gate_1c and gate_1d)
    result = {
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "smoke": args.smoke,
        "entities": {"L0": {"repo": L0.repo_id, "sha": l0_sha},
                     "PG": {"repo": PG.repo_id, "sha": pg_sha}},
        "gate_1a_firewall": {"L0_num_layers": l0_layers, "PG_num_layers": pg_layers, "ok": gate_1a},
        "gate_1b_harness": {"tokenizer_parity": parity["match"],
                            "mmlu": {k: mmlu[k] for k in ("acc_pct", "n", "published", "letter_token_ids")},
                            "mmlu_abs_delta": mmlu_delta, "mmlu_tol": args.tol_mmlu, "mmlu_ok": mmlu_ok,
                            "ppl": ppl["ppl"], "ppl_ok": ppl_ok, "ok": gate_1b},
        "gate_1c_blankimg": {"acc_pct": bi["acc_pct"], "n": bi["n"], "gray": bi["gray"],
                             "logits_finite": logits_finite, "order_match_text": order_match, "ok": gate_1c},
        "gate_1d_item_split": {"n_disc": len(disc), "n_conf": len(conf),
                               "disjoint": disjoint, "ok": gate_1d},
        "build_gate": gate,
        "PREFLIGHT_OK": ok,
    }
    (out / "preflight.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W14-PRE] PREFLIGHT_OK={ok}  (1a={gate_1a} 1b={gate_1b} 1c={gate_1c} 1d={gate_1d})")
    if not ok:
        raise RuntimeError(
            f"W14 PREFLIGHT HALT: firewall={gate_1a} harness={gate_1b} blankimg={gate_1c} "
            f"item_split={gate_1d}. See {out / 'preflight.json'}.")
    print("W14_PREFLIGHT_OK", json.dumps({"mmlu_base_pct": mmlu["acc_pct"],
                                          "blankimg_pct": bi["acc_pct"]}))


if __name__ == "__main__":
    main()
