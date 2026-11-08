"""W11 §9 STEP-1 PREFLIGHT GATE (fail-fast, before any real before→after contrast).

Three sub-gates (prereg §9 step 1). All require GPU (model forwards).

  (1a) Leg-A positive control — base gemma-2-2b reproduces its PUBLISHED MMLU
       (5-shot 51.3) within tol → the text harness is valid; HALT otherwise
       (nothing on the forgetting axis is trustworthy on a broken harness).
       Plus the §8 tokenizer-parity assertion: gemma-2-2b's tokenizer and PG2's
       processor tokenizer encode shared-vocab clean text to identical ids
       (else a spurious perplexity gap). wikitext-2 perplexity recorded as the
       secondary fluency read (no published 2B target → sanity finite only).
  (1b) Leg-B pipeline-port check — reproduce a published per-task PG2 number on
       AWS, INFERENCE-ONLY: AI2D multiple-choice on paligemma2-3b-mix-224
       (published 74.7±0.5, eval_p1) within tol. Validates the PG2-forward +
       battery-style scoring port that Leg B's projector-ablation eval rides on.
       (§9-1b pre-freeze refinement: Leg B is no-training, so the port check is
       inference-only AI2D rather than a *trained* A1 CEILING reproduce — the
       trained reproduce would itself need the LoRA pipeline Leg B never uses.)
  (1c) Split assertion — DM.enforce_split returns a non-empty, disjoint held set
       on a dummy mask (the §9-1c firewall that kills "held captured-but-unused").

Any fail → HALT (raise, no try/except swallowing).

    python -m infra.aws_pg.jobs.w11_preflight --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol-mmlu", type=float, default=3.0,
                    help="abs tol (pct points) on base gemma-2-2b MMLU vs published 51.3")
    ap.add_argument("--tol-ai2d", type=float, default=2.0,
                    help="abs tol (pct points) on PG2-mix AI2D vs published 74.7")
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0,
                    help="cap test items per MMLU subject (0 = full test split)")
    ap.add_argument("--ai2d-n", type=int, default=10**9,
                    help="AI2D test items (clamped to full split)")
    ap.add_argument("--ppl-max-tokens", type=int, default=120_000)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # cut fragmentation

    # MUST precede any huggingface_hub/transformers/datasets import (run_job.sh's
    # HF_HUB_OFFLINE=1 gets cached into a module constant on first hf import →
    # OfflineModeIsEnabled). diff_harvest's top is light (no hf).
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer, PaliGemmaProcessor

    from infra.pg import diff_metrics as DM
    from infra.pg import eval_p1
    from infra.pg import text_eval as TE
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "W11 preflight needs GPU (model forwards) — no CUDA visible."
    device = "cuda"
    print(f"[W11-PRE] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")

    api = HfApi()
    L0 = ENTITIES["L0"]            # google/gemma-2-2b (BEFORE — LM init)
    PG = ENTITIES["PG"]            # google/paligemma2-3b-pt-224 (tokenizer-parity partner)
    l0_sha = api.model_info(L0.repo_id, revision=L0.revision or "main").sha
    pg_sha = api.model_info(PG.repo_id).sha
    print(f"[W11-PRE] L0={L0.repo_id}@{l0_sha}  PG={PG.repo_id}@{pg_sha}")

    # =====================================================================
    # (1a) Leg-A positive control: base gemma-2-2b MMLU + tokenizer parity + ppl.
    # =====================================================================
    model = AutoModelForCausalLM.from_pretrained(
        L0.repo_id, revision=L0.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()  # eager: Gemma2 attn soft-capping
    tok_g = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    tok_pg = PaliGemmaProcessor.from_pretrained(PG.repo_id).tokenizer

    parity = TE.tokenizer_parity(tok_g, tok_pg)
    print(f"[W11-PRE][1a] tokenizer parity match={parity['match']} (n={parity['n']})")

    n_per = args.mmlu_n_per_subject or None
    mmlu = TE.mmlu_accuracy(model, tok_g, device, n_per_subject=n_per)
    mmlu_delta = abs(mmlu["acc_pct"] - mmlu["published"])
    mmlu_ok = bool(mmlu_delta <= args.tol_mmlu)
    print(f"[W11-PRE][1a] MMLU acc={mmlu['acc_pct']:.2f} vs published {mmlu['published']} "
          f"|Δ|={mmlu_delta:.2f} (tol {args.tol_mmlu}) n={mmlu['n']} → ok={mmlu_ok}")

    ppl = TE.clean_text_perplexity(model, tok_g, device, max_tokens=args.ppl_max_tokens)
    # finite-only sanity: ppl is the SECONDARY fluency read with no published 2B target;
    # the absolute is uninterpretable for a 256k-vocab SentencePiece model (high per-token
    # ppl) and the Leg-A decision uses the base-vs-PG2 RATIO (windowing cancels), so the
    # harness-validity gate is MMLU (above), not an absolute ppl threshold.
    ppl_ok = bool(np.isfinite(ppl["ppl"]))
    print(f"[W11-PRE][1a] wikitext ppl={ppl['ppl']:.3f} (finite-only sanity) "
          f"n_tok={ppl['n_tokens']} → ok={ppl_ok}")

    gate_1a = bool(parity["match"] and mmlu_ok and ppl_ok)
    del model
    torch.cuda.empty_cache()

    # =====================================================================
    # (1b) Leg-B pipeline-port: PG2-mix AI2D (inference-only) vs published 74.7.
    # =====================================================================
    ai2d = eval_p1.run_ai2d_eval(n_items=args.ai2d_n, seed=0)
    ai2d_delta = abs(ai2d["acc_exact_pct"] - ai2d["published"])
    gate_1b = bool(ai2d_delta <= args.tol_ai2d)
    print(f"[W11-PRE][1b] AI2D acc={ai2d['acc_exact_pct']:.2f} vs published {ai2d['published']} "
          f"|Δ|={ai2d_delta:.2f} (tol {args.tol_ai2d}) n={ai2d['n_items']} → ok={gate_1b}")

    # =====================================================================
    # (1c) Split firewall — enforce_split on a dummy mask (Leg A text + Leg B task strata).
    # =====================================================================
    rng = np.random.default_rng(1337)
    dummy_held = rng.random(1000) < 0.25
    dummy_text = rng.random(1000) < 0.5
    split = DM.enforce_split(dummy_held, name="preflight", text=dummy_text)
    # disjointness is structural (enforce_split derives disc=~held + asserts internally);
    # the load-bearing gate is both halves AND the text stratum non-empty.
    gate_1c = bool(split["n_conf"] > 0 and split["n_disc"] > 0
                   and split["conf_text_idx"].size > 0)
    print(f"[W11-PRE][1c] enforce_split disc={split['n_disc']} conf={split['n_conf']} "
          f"conf_text={split['conf_text_idx'].size} → ok={gate_1c}")

    ok = bool(gate_1a and gate_1b and gate_1c)
    result = {
        "torch": torch.__version__,
        "entities": {"L0": {"repo": L0.repo_id, "sha": l0_sha},
                     "PG": {"repo": PG.repo_id, "sha": pg_sha},
                     "PG_mix": {"repo": ENTITIES["PG_mix"].repo_id, "ai2d_model": ai2d["model"]}},
        "gate_1a_legA_text": {
            "tokenizer_parity": parity,
            "mmlu": {k: mmlu[k] for k in
                     ("acc_pct", "n", "n_subjects", "n_shots", "n_per_subject",
                      "published", "published_source", "letter_token_ids")},
            "mmlu_abs_delta": mmlu_delta, "mmlu_tol": args.tol_mmlu, "mmlu_ok": mmlu_ok,
            "perplexity": ppl, "ppl_ok": ppl_ok,
            "ok": gate_1a},
        "gate_1b_legB_port": {
            "task": "AI2D", "model": ai2d["model"], "metric": ai2d["metric"],
            "acc_exact_pct": ai2d["acc_exact_pct"], "published": ai2d["published"],
            "abs_delta": ai2d_delta, "tol": args.tol_ai2d, "n_items": ai2d["n_items"],
            "published_source": ai2d["published_source"], "ok": gate_1b},
        "gate_1c_split": {"n": int(split["n"]), "n_disc": int(split["n_disc"]),
                          "n_conf": int(split["n_conf"]), "ok": gate_1c},
        "PREFLIGHT_OK": ok,
    }
    (out / "preflight.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W11-PRE] PREFLIGHT_OK={ok}  (1a={gate_1a} 1b={gate_1b} 1c={gate_1c})")
    if not ok:
        raise RuntimeError(
            f"W11 PREFLIGHT HALT: 1a_legA={gate_1a} (parity={parity['match']} "
            f"mmlu_ok={mmlu_ok} ppl_ok={ppl_ok}) 1b_legB_port={gate_1b} 1c_split={gate_1c}. "
            f"See {out / 'preflight.json'}.")
    print("[W11-PRE] gate passed → safe to develop the W11 behavioral harnesses on control data.")


if __name__ == "__main__":
    main()
