"""Diagnostic (CPU, no GPU, no model weights): how does the PaliGemma processor
tokenize a cloze prompt vs prompt+continuation? Pins the exact prefix/suffix the
processor adds so mmlu_cloze's continuation-span logic is correct on the image path.

    python -m infra.aws_pg.jobs.reprice_proc_probe --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _lcp(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def _lcs(a, b):
    i = 0
    while i < len(a) and i < len(b) and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    from PIL import Image
    from transformers import AutoTokenizer, PaliGemmaProcessor
    from datasets import load_dataset
    from infra.pg.entities import ENTITIES
    from infra.pg.text_eval import _q_block_cloze, _subject_header, MMLU_REPO

    repo = ENTITIES["PG"].repo_id
    proc = PaliGemmaProcessor.from_pretrained(repo)
    tok = proc.tokenizer
    L0 = ENTITIES["L0"]
    tokL0 = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    blank = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))

    ds = load_dataset(MMLU_REPO, "all")
    dev = [r for r in ds["dev"] if r["subject"] == "abstract_algebra"][:5]
    test = [r for r in ds["test"] if r["subject"] == "abstract_algebra"][0]
    prefix = _subject_header("abstract_algebra") + "".join(
        _q_block_cloze(s["question"], s["choices"][s["answer"]]) for s in dev)
    prompt = prefix + _q_block_cloze(test["question"])
    c0 = str(test["choices"][0]).strip()
    full = prompt + " " + c0

    rep = {"image_token_index": getattr(proc, "image_token_index", None),
           "choice0": c0, "choices": test["choices"]}

    # ---- IMAGE path (processor) ----
    pi = proc(text=[prompt], images=[blank], return_tensors="pt", padding="longest")
    fi = proc(text=[full], images=[blank], return_tensors="pt", padding="longest")
    p_ids = pi["input_ids"][0].tolist()
    f_ids = fi["input_ids"][0].tolist()
    # strip image tokens for readability of the text tail
    iti = getattr(proc, "image_token_index", None)
    p_txt = [t for t in p_ids if t != iti]
    f_txt = [t for t in f_ids if t != iti]
    lcp = _lcp(p_txt, f_txt)
    lcs = _lcs(p_txt, f_txt)
    cont_mid = f_txt[lcp: len(f_txt) - lcs]
    rep["image_path"] = {
        "prompt_len_total": len(p_ids), "full_len_total": len(f_ids),
        "n_image_tokens_prompt": p_ids.count(iti), "n_image_tokens_full": f_ids.count(iti),
        "prompt_txt_tail": p_txt[-6:], "full_txt_tail": f_txt[-8:],
        "prompt_txt_tail_decoded": [tok.decode([t]) for t in p_txt[-6:]],
        "full_txt_tail_decoded": [tok.decode([t]) for t in f_txt[-8:]],
        "lcp": lcp, "lcs": lcs,
        "cont_mid_ids": cont_mid, "cont_mid_decoded": [tok.decode([t]) for t in cont_mid],
        "clean_cont_ids": tokL0.encode(" " + c0, add_special_tokens=False),
        "prefix_is_clean": f_txt[:len(p_txt)] == p_txt,
    }

    # ---- TEXT path (tokenizer, sanity) ----
    pt = tokL0(prompt, add_special_tokens=True)["input_ids"]
    ft = tokL0(full, add_special_tokens=True)["input_ids"]
    rep["text_path"] = {"prefix_is_clean": ft[:len(pt)] == pt,
                        "cont_suffix_ids": ft[len(pt):],
                        "cont_suffix_decoded": [tok.decode([t]) for t in ft[len(pt):]]}

    (out / "proc_probe.json").write_text(json.dumps(rep, indent=1, default=str))
    print("[PROBE] image lcp/lcs:", rep["image_path"]["lcp"], rep["image_path"]["lcs"],
          "prefix_clean:", rep["image_path"]["prefix_is_clean"])
    print("[PROBE] full_txt_tail_decoded:", rep["image_path"]["full_txt_tail_decoded"])
    print("[PROBE] cont_mid_decoded:", rep["image_path"]["cont_mid_decoded"])
    print("[PROBE] clean_cont (text tok):", rep["image_path"]["clean_cont_ids"])
    print("[PROBE] text_path prefix_clean:", rep["text_path"]["prefix_is_clean"],
          "cont:", rep["text_path"]["cont_suffix_decoded"])


if __name__ == "__main__":
    main()
