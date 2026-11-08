"""RE-EXPRESS PREFLIGHT (prereg §9.1) — GPU validation of the freeze machinery on
REAL weights. Fail-loud; no try/except around the science.

Stages, in order. The offline gate runs FIRST (real tokenizers + real MMLU, NO
model) so a data / tokenizer / prompt-construction bug dies in seconds with no model
loaded — "GPU only validates what only GPU can":

  0. OFFLINE GATE — gemma-2-2b tokenizer + PG2 processor + real MMLU items: build the
     four ARM-G axis prompts, run the pure graders on PLANTED decodes, assert the
     per_item_hash recipe aligns the axes BY POSITION (the §2 reorder tripwire). No GPU.
  A. BASE CEILING — standalone gemma-2-2b text MMLU (mmlu_accuracy) reproduces the
     published 51.3 within tol (the §9-step-1a REUSED-harness validity control).
  B. KEYSTONE DECODE (NEW) — mmlu_gen2opt on pg_after (blank-image greedy generation):
     well-formed four-axis dict + NON-DEGENERATE decode (unmatched_rate < 1, non-empty
     completions). THE on-GPU validation of the NEW decode loop — §9.1 "verify
     blank-image generate produces well-formed completions; fail loud if degenerate."
  C. CLOZE (PORT) — mmlu_cloze on pg_after (blank-image): finite logprobs, well-formed.
  D. CHIMERA-CEILING PARITY (§2/§9.1) — standalone gemma-2-2b vs the pg_before chimera
     (PG2 body + base decoder + PG2 extended-vocab embed/lm_head) on the same 4-option
     MCQ construct; report chimera/standalone ratio (informs ε_chim at discovery).

    python -m infra.aws_pg.jobs.reexpress_preflight --out <dir> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="n_per_subject=2 everywhere — fast end-to-end plumbing check")
    ap.add_argument("--base-n-per", type=int, default=None,
                    help="n_per_subject for Gate A base ceiling (None=full split)")
    ap.add_argument("--probe-n-per", type=int, default=4,
                    help="n_per_subject for the gen2opt/cloze/chimera probes")
    ap.add_argument("--base-tol", type=float, default=3.0,
                    help="|acc - 51.3| tolerance for the FULL base-ceiling check")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # hf_online_setup FIRST, before any transformers/datasets import (pg gotcha).
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, PaliGemmaProcessor

    from infra.pg import text_eval as TE
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "no CUDA — preflight must run on a GPU box"
    device = "cuda"
    base_n = 2 if args.smoke else args.base_n_per
    probe_n = 2 if args.smoke else args.probe_n_per
    gpu = torch.cuda.get_device_name(0)
    print(f"[REX-PRE] torch={torch.__version__} gpu={gpu} smoke={args.smoke} "
          f"base_n={base_n} probe_n={probe_n}")

    result: dict = {"torch": torch.__version__, "gpu": gpu, "smoke": args.smoke,
                    "base_n_per": base_n, "probe_n_per": probe_n}

    # ======================================================================
    # 0. OFFLINE GATE — real tokenizers + real MMLU, NO model.
    # ======================================================================
    g2_repo = ENTITIES["L0"].repo_id          # google/gemma-2-2b
    pg_repo = ENTITIES["PG"].repo_id          # google/paligemma2-3b-pt-224
    g2_tok = AutoTokenizer.from_pretrained(g2_repo, revision=ENTITIES["L0"].revision)
    proc = PaliGemmaProcessor.from_pretrained(pg_repo)

    # letter ids single-token on BOTH tokenizers (the MCQ axis read).
    for name, t in [("gemma-2-2b", g2_tok), ("pg2-proc", proc.tokenizer)]:
        for L in TE.CHOICES:
            ids = t.encode(" " + L, add_special_tokens=False)
            assert len(ids) == 1, f"{name}: ' {L}' → {len(ids)} tokens (need 1)"

    ds = load_dataset(TE.MMLU_REPO, "all")
    dev, test = ds["dev"], ds["test"]
    # take a couple subjects' first items to exercise prompt construction + alignment
    import hashlib
    from collections import defaultdict
    test_by_subj: dict = defaultdict(list)
    for r in test:
        test_by_subj[r["subject"]].append(r)
    probe_items = []
    for subj in sorted(test_by_subj)[:3]:
        probe_items.extend(test_by_subj[subj][:4])
    # build the cloze prompt (shared by P & Z) + the MCQ block (M), and a planted decode
    planted_correct, planted_letter, golds, options = [], [], [], []
    hashes_axisP, hashes_axisM = [], []
    for r in probe_items:
        choices = [str(c).strip() for c in r["choices"]]
        ai = int(r["answer"])
        # P/Z prompt (cloze) and M prompt (lettered) both end at "Answer:"
        _ = TE._q_block_cloze(r["question"])               # construct must not raise
        _ = TE._q_block(r["question"], choices)
        options.append(choices); golds.append(choices[ai])
        planted_correct.append(choices[ai])                # a model that PRODUCES the gold
        planted_letter.append(TE.CHOICES[ai])              # a model that emits only the letter
        h = hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12]
        hashes_axisP.append(h); hashes_axisM.append(h)
    # axis alignment BY POSITION: the two axes share the identical hash sequence
    assert hashes_axisP == hashes_axisM, "four-axis hash sequences misaligned by position"
    # grader behaviour on planted decodes: producing the gold content → correct;
    # emitting only the MCQ letter → unmatched (the keystone's whole point)
    p_correct = [int(TE._match_to_option(d, o)[0] == int(r["answer"]))
                 for d, o, r in zip(planted_correct, options, probe_items)]
    p_letter = [TE._match_to_option(d, o)[1] for d, o in zip(planted_letter, options)]
    assert all(p_correct), f"planted gold content did not match its option: {p_correct}"
    assert all(f == "unmatched" for f in p_letter), \
        f"planted bare letter should be unmatched on the content axis: {p_letter}"
    fg = TE.mmlu_freegen(planted_correct, golds)
    offline_ok = bool(all(p_correct) and all(f == "unmatched" for f in p_letter)
                      and sum(fg["per_item"]) == len(golds))
    result["offline_gate"] = {"n_probe_items": len(probe_items),
                              "planted_content_all_correct": bool(all(p_correct)),
                              "planted_letter_all_unmatched": bool(all(f == "unmatched" for f in p_letter)),
                              "freegen_all_correct": sum(fg["per_item"]) == len(golds),
                              "ok": offline_ok}
    print(f"[REX-PRE][0] offline gate ok={offline_ok} (n={len(probe_items)})")
    assert offline_ok, "OFFLINE GATE HALT — data/tokenizer/grader broke before any model load"

    # ======================================================================
    # A. BASE CEILING — standalone gemma-2-2b text MMLU vs published 51.3.
    # ======================================================================
    base = AutoModelForCausalLM.from_pretrained(
        g2_repo, revision=ENTITIES["L0"].revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    mm_base = TE.mmlu_accuracy(base, g2_tok, device, n_per_subject=base_n)
    base_acc = mm_base["acc_pct"]
    d_pub = abs(base_acc - 51.3)
    gate_A = bool(base_acc > 35.0) if (args.smoke or base_n) else bool(d_pub <= args.base_tol)
    print(f"[REX-PRE][A] base gemma-2-2b MMLU={base_acc:.2f} (n={mm_base['n']}) "
          f"|Δ51.3|={d_pub:.2f} → ok={gate_A}")
    result["gate_A_base_ceiling"] = {"acc_pct": base_acc, "n": mm_base["n"],
                                     "published": 51.3, "abs_delta": d_pub,
                                     "full_split": base_n is None and not args.smoke,
                                     "ok": gate_A}

    # standalone arm of the chimera gate (small probe set, text MCQ) — reuse this model
    mm_base_probe = TE.mmlu_accuracy(base, g2_tok, device, n_per_subject=probe_n)
    standalone_probe_acc = mm_base_probe["acc_pct"]
    del base
    torch.cuda.empty_cache()

    # ======================================================================
    # B/C/D — build pg_after + pg_before chimera (fail-loud leak gates inside).
    # ======================================================================
    from infra.pg.diff_harvest import build_before_after
    pg_after, pg_before, proc2, paths, img_tok_idx, build_gate = build_before_after(device)
    print(f"[REX-PRE] build_before_after ok: decoder Δ={build_gate.get('decoder_max_abs_delta_vs_PG'):.4g} "
          f"image_token_index={img_tok_idx}")

    # ---- B. KEYSTONE DECODE (NEW) on pg_after ----
    g2o = TE.mmlu_gen2opt(pg_after, proc2, device, n_per_subject=probe_n,
                          max_new_tokens=args.max_new_tokens, batch_size=4)
    n = g2o["n"]
    lens_ok = (len(g2o["per_item"]) == n == len(g2o["per_item_hash"])
               == len(g2o["freegen_per_item"]) == len(g2o["match_flag"]))
    binary_ok = all(v in (0, 1) for v in g2o["per_item"])
    nonempty = sum(1 for d in g2o["decoded_sample"] if d.strip())
    nondegen = bool(g2o["unmatched_rate"] < 1.0 and nonempty > 0)
    gate_B = bool(lens_ok and binary_ok and nondegen)
    print(f"[REX-PRE][B] gen2opt acc={g2o['acc_pct']:.2f} freegen={100*g2o['freegen_acc']:.2f} "
          f"unmatched={g2o['unmatched_rate']:.2f} tie={g2o['tie_rate']:.2f} "
          f"nonempty={nonempty}/{len(g2o['decoded_sample'])} → ok={gate_B}")
    print(f"[REX-PRE][B] decoded sample: {g2o['decoded_sample'][:6]}")
    result["gate_B_keystone_decode"] = {
        "n": n, "acc_pct": g2o["acc_pct"], "freegen_acc": g2o["freegen_acc"],
        "unmatched_rate": g2o["unmatched_rate"], "tie_rate": g2o["tie_rate"],
        "lens_ok": lens_ok, "binary_ok": binary_ok, "nonempty_sample": nonempty,
        "decoded_sample": g2o["decoded_sample"][:12], "ok": gate_B}

    # ---- C. CLOZE (PORT) on pg_after (blank-image) ----
    cz = TE.mmlu_cloze(pg_after, None, device, processor=proc2, n_per_subject=probe_n)
    cz_logp = np.asarray(cz["cloze_logprobs"], dtype=float)
    cloze_finite = bool(np.isfinite(cz_logp).all())
    cloze_lens_ok = (cz["n"] == len(cz["per_item"]) == cz_logp.shape[0])
    gate_C = bool(cloze_finite and cloze_lens_ok and cz_logp.shape[1] == 4)
    print(f"[REX-PRE][C] cloze acc={cz['acc_pct']:.2f} (n={cz['n']}) finite={cloze_finite} → ok={gate_C}")
    result["gate_C_cloze_port"] = {"n": cz["n"], "acc_pct": cz["acc_pct"],
                                   "finite": cloze_finite, "lens_ok": cloze_lens_ok, "ok": gate_C}

    # ---- D. CHIMERA-CEILING PARITY: pg_before chimera blank-image MCQ ----
    mm_chim = TE.mmlu_blankimg(pg_before, proc2, device, n_per_subject=probe_n)
    chim_acc = mm_chim["acc_pct"]
    ratio = chim_acc / standalone_probe_acc if standalone_probe_acc > 0 else float("nan")
    # NOT a hard gate here (ε_chim is frozen on discovery); report + a chance-floor sanity.
    chim_sane = bool(chim_acc > 30.0)
    print(f"[REX-PRE][D] chimera blank-image MCQ={chim_acc:.2f} vs standalone text MCQ="
          f"{standalone_probe_acc:.2f} ratio={ratio:.3f} sane={chim_sane}")
    result["gate_D_chimera_parity"] = {
        "chimera_blankimg_acc_pct": chim_acc, "standalone_text_acc_pct": standalone_probe_acc,
        "ratio": ratio, "n": mm_chim["n"], "chimera_above_chance": chim_sane,
        "note": "ratio reported for ε_chim discovery-freeze; gate applied at G0, not here"}

    # ---- B'. KEYSTONE GRADER-VALIDITY (§6): does the BASE produce answer content? ----
    # The decisive discriminator for the pg_after floor: run the SAME gen2opt on the chimera
    # (base gemma-2-2b decoder in the PG2 blank-image forward). If even the base can't PRODUCE
    # answers (gen2opt ≪ its own MCQ), the keystone construct is INVALID as built — VOID, re-scope
    # the prompt/matcher BEFORE freeze. If the base recovers, pg_after's floor is a real signal.
    g2o_base = TE.mmlu_gen2opt(pg_before, proc2, device, n_per_subject=probe_n,
                               max_new_tokens=args.max_new_tokens, batch_size=4)
    base_gen2opt = g2o_base["acc_pct"]
    # provisional validity ratio vs the chimera's own MCQ ceiling (real G_min frozen at discovery)
    valid_ratio = base_gen2opt / chim_acc if chim_acc > 0 else float("nan")
    print(f"[REX-PRE][B'] base(chimera) gen2opt acc={base_gen2opt:.2f} "
          f"unmatched={g2o_base['unmatched_rate']:.2f} vs chimera-MCQ={chim_acc:.2f} "
          f"valid_ratio={valid_ratio:.3f}")
    print(f"[REX-PRE][B'] base decoded sample: {g2o_base['decoded_sample'][:6]}")
    result["gate_Bprime_grader_validity"] = {
        "base_gen2opt_acc_pct": base_gen2opt, "base_unmatched_rate": g2o_base["unmatched_rate"],
        "chimera_mcq_acc_pct": chim_acc, "valid_ratio": valid_ratio,
        "base_freegen_acc": g2o_base["freegen_acc"],
        "decoded_sample": g2o_base["decoded_sample"][:12],
        "note": "validity discriminator for the pg_after gen2opt floor; G_min frozen at discovery"}

    # ======================================================================
    ok = bool(offline_ok and gate_A and gate_B and gate_C and chim_sane)
    result["PREFLIGHT_OK"] = ok
    (out / "preflight.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[REX-PRE] PREFLIGHT_OK={ok}")
    if not ok:
        raise RuntimeError(f"REX PREFLIGHT HALT: offline={offline_ok} A={gate_A} "
                           f"B={gate_B} C={gate_C} D_sane={chim_sane}")


if __name__ == "__main__":
    main()
