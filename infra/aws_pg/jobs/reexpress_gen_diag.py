"""RE-EXPRESS gen-readout DIAGNOSTIC (post-smoke2 keystone-VOID).

smoke2 proved the gen-into-4-options keystone is VOID on the BLANK-IMAGE forward: the
base gemma-2-2b decoder (48% MCQ knowledge) produces 0% answer content, decoded_raw is
pure document-continuation / degeneration (case b/c, NOT a missed-match a) — so a matcher
fix is dishonest. This job runs the user's path-1 pivot: does the TEXT-ONLY forward (no
256 blank-image tokens) recover production on the SAME model?

Standalone gemma-2-2b only (no chimera build). DISCOVERY items only (grader tuning stays
on discovery, §6). Reports text-only gen2opt grader-validity vs the MCQ ceiling on the
SAME items, and DUMPS gold + generation per item for an airtight a/b/c re-call.

  python -m infra.aws_pg.jobs.reexpress_gen_diag --out <dir> [--n-per 2] [--dump 24]
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _text_generate(model, tok, device, prompts, max_new_tokens=16, batch_size=8):
    """TEXT-ONLY greedy decode (no image): left-pad, generate do_sample=False, decode the
    new tokens, truncate at first newline. Aligned 1:1 to `prompts`."""
    import torch
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    orig = tok.padding_side
    tok.padding_side = "left"
    out = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        in_len = enc["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        dec = tok.batch_decode(gen[:, in_len:], skip_special_tokens=True)
        out.extend(d.split("\n")[0].strip() for d in dec)
    tok.padding_side = orig
    assert len(out) == len(prompts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per", type=int, default=2, help="rows/subject BEFORE the discovery filter")
    ap.add_argument("--dump", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import hashlib
    from collections import defaultdict

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as W
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available()
    device = "cuda"
    N = args.n_per
    g2_repo = ENTITIES["L0"].repo_id
    tok = AutoTokenizer.from_pretrained(g2_repo, revision=ENTITIES["L0"].revision)
    base = AutoModelForCausalLM.from_pretrained(
        g2_repo, revision=ENTITIES["L0"].revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    print(f"[GEN-DIAG] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} N={N}")

    def h(q):
        return hashlib.sha1(q[:64].encode()).hexdigest()[:12]

    ds = load_dataset(TE.MMLU_REPO, "all")
    dev, test = ds["dev"], ds["test"]
    shots_by_subj = defaultdict(list)
    for r in dev:
        shots_by_subj[r["subject"]].append(r)
    test_by_subj = defaultdict(list)
    for r in test:
        test_by_subj[r["subject"]].append(r)

    # EXACT mmlu_choice_logits item selection: slice n_per_subject FIRST, THEN filter discovery,
    # so the gen items and the MCQ-ceiling items are identical and position-aligned.
    prompts, golds, options, answers, hashes, qstr = [], [], [], [], [], []
    for subj in sorted(test_by_subj):
        shots = shots_by_subj[subj][:5]
        prefix = TE._subject_header(subj) + "".join(
            TE._q_block_cloze(s["question"], s["choices"][s["answer"]]) for s in shots)
        rows = test_by_subj[subj]
        if N:
            rows = rows[:N]
        rows = [r for r in rows if W.is_discovery(h(r["question"]))]
        for r in rows:
            ch = [str(c).strip() for c in r["choices"]]
            ai = int(r["answer"])
            prompts.append(prefix + TE._q_block_cloze(r["question"]))
            options.append(ch); golds.append(ch[ai]); answers.append(ai)
            hashes.append(h(r["question"])); qstr.append(r["question"])
    assert prompts, "no discovery items selected"
    print(f"[GEN-DIAG] {len(prompts)} discovery items")

    # ---- TEXT-ONLY generation (path-1 production readout) ----
    decoded = _text_generate(base, tok, device, prompts, max_new_tokens=args.max_new_tokens)
    per_item, flags = [], []
    for d, o, ai in zip(decoded, options, answers):
        idx, flag = TE._match_to_option(d, o)
        per_item.append(int(idx == ai)); flags.append(flag)
    n = len(per_item)
    gen_acc = 100 * sum(per_item) / n
    unmatched = sum(f == "unmatched" for f in flags) / n
    free = [TE._free_gen_em(d, g) for d, g in zip(decoded, golds)]
    free_acc = 100 * sum(free) / n

    # ---- MCQ ceiling on the SAME items (alignment asserted) ----
    mcq = TE.mmlu_choice_logits(base, tok, device, n_per_subject=N, item_filter=W.is_discovery)
    assert mcq["per_item_hash"] == hashes, \
        f"MCQ/gen item misalignment: {len(mcq['per_item_hash'])} vs {len(hashes)}"
    mcq_acc = mcq["acc_pct"]
    valid_ratio = gen_acc / mcq_acc if mcq_acc > 0 else float("nan")

    print(f"[GEN-DIAG] TEXT-ONLY gen2opt acc={gen_acc:.2f}  freegen={free_acc:.2f}  "
          f"unmatched={unmatched:.2f}  MCQ-ceiling={mcq_acc:.2f}  valid_ratio={valid_ratio:.3f}")
    print(f"[GEN-DIAG] (smoke2 blank-image base gen2opt was 0.00 / unmatched 0.97 — contrast)")
    print("[GEN-DIAG] ---- decoded_raw dump (gold || generation || flag) ----")
    dump = []
    for i in range(min(args.dump, n)):
        row = {"q": qstr[i][:90], "gold": golds[i], "options": options[i],
               "decoded": decoded[i], "flag": flags[i], "strict_correct": per_item[i],
               "mcq_correct": int(mcq["per_item"][i])}
        dump.append(row)
        print(f"  [{i:2d}] gold={golds[i]!r:40s} gen={decoded[i]!r:50s} {flags[i]} ok={per_item[i]}")

    result = {
        "gpu": torch.cuda.get_device_name(0), "n_items": n, "n_per": N,
        "forward": "text-only", "model": g2_repo,
        "textonly_gen2opt_acc_pct": gen_acc, "textonly_freegen_acc_pct": free_acc,
        "textonly_unmatched_rate": unmatched, "mcq_ceiling_acc_pct": mcq_acc,
        "valid_ratio": valid_ratio,
        "smoke2_blankimage_base_gen2opt_acc_pct": 0.0,
        "decoded_dump": dump,
        "interpretation": ("valid_ratio≥~0.8 → text-only IS the production axis (path-1 works, "
                           "re-scope keystone to text-only forward); valid_ratio≈0 → free-gen is "
                           "intrinsically invalid for base gemma → path-2 causal 'use' proxy"),
    }
    (out / "gen_diag.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[GEN-DIAG] valid_ratio={valid_ratio:.3f} → "
          f"{'TEXT-ONLY VIABLE (path-1)' if valid_ratio >= 0.5 else 'free-gen INVALID (path-2)'}")


if __name__ == "__main__":
    main()
