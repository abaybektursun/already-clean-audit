"""LACUNA Arm 1 extension — the FORCED-CHOICE multimodal readout (POST-HOC, EXPLORATORY).

Why this exists. The frozen mm ladder scores free generation, and the π0 re-shell generates
rare-token gibberish: VQAv2 soft accuracy came back at EXACTLY 0.0 for both the real-image and the
gray-image condition. When both conditions sit at a hard 0, **Δ_perception — the spec's one
discriminator between H_uniform (the capability was consumed) and H_shell (the re-shell forward is
OOD) — is identically zero and adjudicates nothing.** The generation readout has no range at the
floor, so it cannot answer the question the program was built to ask.

A forced choice does. Per item, score the summed log-probability of the GOLD answer string and of
three in-distribution distractor answers as continuations of the same image+question prompt, and
take the length-normalised argmax. Properties that matter here:

  * the floor is **0.25 by construction** (exact chance, not an empirical band that can saturate);
  * it needs no parseable generation, so a broken output distribution still yields a ranking;
  * paired {real, gray} gives a Δ_perception WITH range even when both generation scores are 0.

STATUS AND SCOPE. This is added AFTER seeing that the frozen readout floored, so it is
**exploratory and post-hoc**, is NOT in the confirmatory family, and does not modify or replace any
frozen verdict (`feedback_no_post_unblind_rule_swap`). It is reported as its own labelled arm. What
it can do is turn "we could not tell" into a measurement: if π0 sits at chance on BOTH conditions,
"nothing is readable" is established on a readout that had range; if it beats chance with
real > gray, image information demonstrably flows and H_uniform is wrong — a finding the frozen
readout was structurally unable to see.

Scoring core REUSED verbatim from the house cloze instrument
(`text_eval._common_prefix_len/_common_suffix_len/_score_cloze_window`), including its PaliGemma
trailing-'\\n' framing assertions.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.lacuna_mmcloze --type g6e.2xlarge --no-weights \\
      --args "--rows pg1,matched_dw,pi0,pi05 --n 1500" --s3-prefix results/lacuna_mmcloze \\
      --budget-min 420
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

N_CAND = 4
PPL_ANCHOR = 22.42583
PPL_REL_TOL = 1e-3


def build_choices(rows, task, seed):
    """(candidates, gold_idx) per item: the majority annotator answer plus N_CAND-1 distractors
    drawn from the corpus's own frequent-answer pool, excluding every answer this item's
    annotators gave (so a distractor is never silently correct)."""
    import numpy as np
    from collections import Counter
    golds, all_ans = [], Counter()
    for r in rows:
        ans = [a["answer"].strip().lower() for a in r["answers"]]
        all_ans.update(ans)
        golds.append(Counter(ans).most_common(1)[0][0])
    pool = [a for a, _ in all_ans.most_common(200)]
    rng = np.random.default_rng(seed)
    out = []
    for r, g in zip(rows, golds):
        banned = {a["answer"].strip().lower() for a in r["answers"]} | {g}
        cand_pool = [a for a in pool if a not in banned]
        assert len(cand_pool) >= N_CAND - 1, "distractor pool exhausted"
        d = list(rng.choice(cand_pool, size=N_CAND - 1, replace=False))
        opts = d + [g]
        order = rng.permutation(N_CAND)
        opts = [opts[i] for i in order]
        out.append((opts, opts.index(g)))
    return out


def score_item(model, proc, device, image, prompt, cands):
    """(summed logprob per candidate, ntok per candidate) for `cands` continuing `prompt` with
    `image` present. Mirrors `text_eval.mmlu_cloze`'s image path exactly."""
    import torch
    from infra.pg.text_eval import (
        _common_prefix_len, _common_suffix_len, _score_cloze_window,
    )
    tok = proc.tokenizer
    orig = tok.padding_side
    tok.padding_side = "left"
    pe = proc(text=[prompt], images=[image], return_tensors="pt", padding="longest").to(device)
    prompt_real = pe["input_ids"][0][int((pe["attention_mask"][0] == 0).sum()):].tolist()
    enc = proc(text=[prompt + " " + c for c in cands], images=[image] * len(cands),
               return_tensors="pt", padding="longest").to(device)
    ids, am = enc["input_ids"], enc["attention_mask"]
    real = am.sum(dim=1).tolist()
    cont_ids, ntok, lcp, lcs = [], [], None, None
    for k in range(len(cands)):
        full_real = ids[k][ids.shape[1] - int(real[k]):].tolist()
        p = _common_prefix_len(prompt_real, full_real)
        s = _common_suffix_len(prompt_real, full_real)
        assert p + s == len(prompt_real), f"prompt not [shared|sep]: lcp={p} lcs={s}"
        cont = full_real[p:len(full_real) - s]
        assert len(cont) >= 1, "empty continuation"
        if lcp is None:
            lcp, lcs = p, s
        else:
            assert (p, s) == (lcp, lcs), "inconsistent framing across candidates"
        cont_ids.append(cont)
        ntok.append(len(cont))
    K = max(ntok) + lcs + 1
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=am,
                    pixel_values=enc["pixel_values"].to(torch.bfloat16), logits_to_keep=K)
        lw = torch.log_softmax(out.logits.float(), dim=-1)
    sums = _score_cloze_window(lw, cont_ids, ntok, trailing=lcs)
    tok.padding_side = orig
    assert torch.isfinite(sums).all(), "non-finite candidate logprob"
    return [float(x) for x in sums.tolist()], [int(m) for m in ntok]


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="pg1,matched_dw,pi0,pi05")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--n-pool", type=int, default=4000,
                    help="items streamed before subsetting (keeps the ladder's item order)")
    ap.add_argument("--conditions", default="real,gray")
    ap.add_argument("--choice-seed", type=int, default=20260728)
    ap.add_argument("--matched-dw-seed", type=int, default=20260727)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [r.strip() for r in a.rows.split(",") if r.strip()]
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_shell as LS

    assert torch.cuda.is_available(), "lacuna_mmcloze needs GPU"
    device = "cuda"
    t0 = time.time()
    rep = {"object": "lacuna_mmcloze_v1", "rows": rows, "conditions": conds, "n": a.n,
           "n_cand": N_CAND, "chance": 1.0 / N_CAND, "gpu": torch.cuda.get_device_name(0),
           "label": "POST-HOC EXPLORATORY — not in the confirmatory family; added because the "
                    "frozen generation readout floored at exactly 0.0 for the policy rows, which "
                    "makes Δ_perception identically zero and unable to separate H_uniform from "
                    "H_shell",
           "seeds": {"choice": a.choice_seed, "matched_dw": a.matched_dw_seed}}
    cells, banks = [], {}
    rep["cells"] = cells

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
        (out / "mmcloze_banks.json").write_text(json.dumps(banks, default=str))
    _w()

    tasks = LS.mm_tasks()
    vqa = tasks["vqav2"]
    pool_rows, pool_hashes = IP.load_items(vqa, a.n_pool)
    items = pool_rows[: a.n]
    hashes = pool_hashes[: a.n]
    choices = build_choices(items, vqa, a.choice_seed)
    banks["item_hashes"] = hashes
    banks["gold_idx"] = [g for _c, g in choices]
    rep["items"] = {"n": len(items), "n_pool": a.n_pool,
                    "note": "the ladder's own streaming order and content hashes — the same "
                            "discovery/confirmation split applies"}
    _w()

    model, proc = LS.load_pg1(device=device)
    pg1_sd = LS.pg1_baseline_sd(model)
    ppl = IP.ppl_gate(model, proc, device)
    val = float(ppl["ppl"] if isinstance(ppl, dict) else ppl)
    rel = abs(val - PPL_ANCHOR) / PPL_ANCHOR
    rep["G2_ppl_anchor"] = {"ppl": val, "rel_diff": rel, "passes": rel < PPL_REL_TOL,
                            "gpu": torch.cuda.get_device_name(0)}
    assert rep["G2_ppl_anchor"]["passes"], f"G2 FAIL: {val} (rel {rel:.2e})"
    _w()

    pol_cache = {}

    def _policy(nm):
        if nm not in pol_cache:
            for old in list(pol_cache):
                del pol_cache[old]
            import gc
            gc.collect()
            pol_cache[nm] = LS.policy_full_sd(nm)
        return pol_cache[nm]

    def build(row):
        LS.restore_pg1(model, pg1_sd)
        if row == "pg1":
            return {"row": "pg1"}
        if row in ("pi0", "pi05"):
            sd, sha = _policy(row)
            return {**LS.build_row_policy(model, row, embed="own", full_sd=sd), "sha": sha}
        if row == "matched_dw":
            sd, _ = _policy("pi0")
            lm, _h, _c = LS.policy_vlm_sd("pi0", full_sd=sd)
            target = {k[len("language_model."):]: v for k, v in lm.items()
                      if k.startswith("language_model.")}
            return LS.build_row_matched_dw(model, pg1_sd, target, a.matched_dw_seed)
        raise RuntimeError(f"unknown row {row}")

    from infra.aws_pg.jobs.invariant_core_func import _blank_image
    gray_img = _blank_image(128)

    for row in rows:
        info = build(row)
        rep.setdefault("row_info", {})[row] = info
        print(f"[MMC] built {row}", flush=True)
        _w()
        for cond in conds:
            lps, nts, per_item = [], [], []
            for i, (r, (cands, gold)) in enumerate(zip(items, choices)):
                img = vqa.image_fn(r).convert("RGB") if cond == "real" else gray_img
                s, nt = score_item(model, proc, device, img, vqa.prompt_fn(r), cands)
                lps.append(s); nts.append(nt)
                mean_lp = np.asarray(s) / np.asarray(nt, dtype=float)
                per_item.append(int(int(mean_lp.argmax()) == gold))
                if (i + 1) % 250 == 0:
                    print(f"[MMC] {row}/{cond} {i+1}/{len(items)} "
                          f"acc={np.mean(per_item):.4f}", flush=True)
            acc = float(np.mean(per_item))
            cells.append({"row": row, "cond": cond, "acc": acc, "n": len(per_item),
                          "chance": 1.0 / N_CAND, "elapsed_s": round(time.time() - t0, 1)})
            banks[f"{row}|{cond}"] = {"per_item": per_item, "logprobs": lps, "ntoks": nts}
            print(f"[MMC] {row:>12s} {cond:>5s} acc={acc:.4f} (chance {1/N_CAND})", flush=True)
            _w()

    rep["elapsed_s"] = round(time.time() - t0, 1)
    _w()
    print("LACUNA_MMCLOZE_OK", json.dumps({f"{c['row']}|{c['cond']}": c["acc"] for c in cells}),
          flush=True)


if __name__ == "__main__":
    main()
