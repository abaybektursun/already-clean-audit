"""LACUNA Arm 1 — the TEXT half of the ladder (spec §2.1; fills matrix cell 4A-cloze).

The cloze/RFP readouts exist for {base, floor, π0, π0.5, matched-ΔW} (PCR 2026-06-19, RFP
2026-06-20) but SKIP PG1 — so the deployed policies have never been scored against the stage they
actually came from. This job runs the identical readouts with PG1's own LM decoder as a row, plus
every control row, in ONE pass so the whole text ladder is internally consistent and nothing is
transported from a prior artifact.

Rows (decoder → text shell; every builder REUSED):
  base        base gemma-2b                                    (the pure-prior reference)
  pg1         PG1's 18L LM decoder, base-embed shell           **CEILING** (the NEW row)
  pi0 / pi05  policy decoder under its OWN tied head           (primary)
  pi0_baseembed / pi05_baseembed   policy decoder, base embed  (interface control)
  matched_dw  PG1 decoder + Frobenius-matched Gaussian ΔW of the **PG1→π0** hop  **FLOOR**
  rand_dec    PG1 decoder → std-matched Gaussian                (scale zero, ordering check)

Axes: MMLU-MCQ letters · MMLU cloze content · read-from-passage (cloze + MCQ) · wikitext ppl.
All four are per-item banked; every band, S and label is recomputed downstream by
`lacuna_aggregate` from the banks.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.lacuna_text --type g6e.8xlarge --no-weights \
      --args "--rows base,pg1,matched_dw" --s3-prefix results/lacuna_text_a --budget-min 300
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

ROW_ORDER = ("base", "pg1", "matched_dw", "rand_dec", "pi0", "pi05",
             "pi0_baseembed", "pi05_baseembed")
BASEEMBED_OF = {"pi0_baseembed": "pi0", "pi05_baseembed": "pi05"}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(ROW_ORDER))
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0, help="0 = full n≈14,042 bank")
    ap.add_argument("--rfp-n", type=int, default=2000)
    ap.add_argument("--rfp-seed", type=int, default=20260620)
    ap.add_argument("--ppl-max-tokens", type=int, default=120_000)
    ap.add_argument("--matched-dw-seed", type=int, default=20260727)
    ap.add_argument("--rand-dec-seed", type=int, default=1337)
    ap.add_argument("--axes", default="mcq,cloze,rfp,ppl")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in ROW_ORDER if r in {x.strip() for x in a.rows.split(",")}]
    axes = [x.strip() for x in a.axes.split(",") if x.strip()]
    assert rows, f"no known rows in {a.rows!r}"

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from infra.aws_pg.jobs.policy_content_readout import (
        _build_ownembed_model, _extract_own, matched_dw_decoder, vocab_parity_gate,
    )
    from infra.aws_pg.jobs.vlat_retention import _extract_decoder, _random_decoder_sd
    from infra.pg import read_from_passage as RFP
    from infra.pg import text_eval as TE
    from infra.pg import vlat_swapback as SB
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "lacuna_text needs GPU"
    device = "cuda"
    t0 = time.time()
    L0 = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    n_per = a.mmlu_n_per_subject or None

    rep = {"object": "lacuna_text_v1", "rows": rows, "axes": axes,
           "mmlu_n_per_subject": n_per, "rfp_n": a.rfp_n, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0),
           "seeds": {"matched_dw": a.matched_dw_seed, "rand_dec": a.rand_dec_seed,
                     "rfp": a.rfp_seed}}
    banks, meta = {}, {}
    rep["meta"], rep["acc"] = meta, {}
    _acc = rep["acc"]

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
        (out / "lacuna_text_banks.json").write_text(json.dumps(banks, default=str))
    _w()

    # ---- weight-free gate: every cloze candidate resolves in the shared base vocab ------
    rep["vocab_parity"] = vocab_parity_gate(tok, n_per_subject=(n_per or 2))
    _w()

    # ---- the RFP item set (frozen synthetic bank; content-hashed) -----------------------
    rfp_items, rfp_ex = None, None
    if "rfp" in axes:
        built = RFP.build_synth(a.rfp_seed, a.rfp_n)
        rfp_items, rfp_ex = built[0], built[1]
        rep["rfp"] = {"n": len(rfp_items), "content_hash": RFP.content_hash(rfp_items),
                      "n_exemplars": len(rfp_ex)}
        _w()

    # decoders staged for the derived rows (matched_dw / rand_dec need PG1's + π0's)
    stash = {}

    def _decoder(name):
        if name not in stash:
            dec, sha = _extract_decoder(name, device, "")
            stash[name] = ({k: v.detach().cpu().clone() for k, v in dec.items()} if dec else None,
                           sha)
        return stash[name]

    def build(row):
        if row == "base":
            m = AutoModelForCausalLM.from_pretrained(
                L0.repo_id, revision=L0.revision, torch_dtype=torch.bfloat16,
                device_map=device, attn_implementation="eager").eval()
            return m, {"readout": "base gemma-2b (base embed) — pure-prior reference"}
        if row == "pg1":
            dec, sha = _decoder("PG1")
            m, info = SB.load_swapback_model(dec, device)
            return m, {"readout": "PG1 LM decoder, base-embed shell — CEILING", "sha": sha, **info}
        if row == "matched_dw":
            pg1_dec, _ = _decoder("PG1")
            pi0_dec, _ = _decoder("pi0")
            mdec, frob = matched_dw_decoder(pg1_dec, pi0_dec, a.matched_dw_seed)
            m, info = SB.load_swapback_model(mdec, device)
            tot = sum(v * v for v in frob.values()) ** 0.5
            return m, {"readout": "PG1 decoder + Frobenius-matched Gaussian ΔW of the PG1→π0 hop "
                                  "— FLOOR (AMENDMENT-1)", "total_frob_dW": float(tot),
                       "n_matched": len(frob), **info}
        if row == "rand_dec":
            pg1_dec, _ = _decoder("PG1")
            m, info = SB.load_swapback_model(_random_decoder_sd(pg1_dec, a.rand_dec_seed), device)
            return m, {"readout": "PG1 decoder → std-matched Gaussian (scale zero)", **info}
        if row in BASEEMBED_OF:
            pol = BASEEMBED_OF[row]
            dec, _own, sha = _extract_own(pol, device)
            m, info = SB.load_swapback_model(dec, device)
            return m, {"readout": f"{pol} decoder + BASE embed (interface control)",
                       "sha": sha, **info}
        dec, own_head, sha = _extract_own(row, device)
        m, info = _build_ownembed_model(dec, own_head, device)
        return m, {"readout": f"{row} own-embed (tied lm_head)", "sha": sha, **info}

    for row in rows:
        model, info = build(row)
        meta[row] = info
        print(f"[LTX] {row}: {json.dumps(info, default=str)[:220]}", flush=True)
        b = {}
        banks[row] = b               # bound BEFORE the axes (fleet lesson): a crash mid-row keeps
        _w()                         # whichever axes already finished
        if "mcq" in axes:
            mc = TE.mmlu_choice_logits(model, tok, device, n_per_subject=n_per)
            b["mcq"] = {"per_item": mc["per_item"], "hash": mc["per_item_hash"],
                        "acc_pct": mc["acc_pct"], "n": mc["n"]}
            _acc.setdefault("mmlu_mcq", {})[row] = mc["acc_pct"]
            print(f"[LTX] {row} MMLU-MCQ={mc['acc_pct']:.2f} (n={mc['n']})", flush=True)
            _w()
        if "cloze" in axes:
            cz = TE.mmlu_cloze(model, tok, device, n_per_subject=n_per)
            b["cloze"] = {"logprobs": cz["cloze_logprobs"], "ntoks": cz["cloze_ntoks"],
                          "correct_idx": cz["correct_idx"], "hash": cz["per_item_hash"],
                          "acc_pct": cz["acc_pct"], "n": cz["n"]}
            _acc.setdefault("mmlu_cloze", {})[row] = cz["acc_pct"]
            print(f"[LTX] {row} MMLU-CLOZE={cz['acc_pct']:.2f} (n={cz['n']})", flush=True)
            _w()
        if "rfp" in axes:
            rc = RFP.passage_cloze(model, tok, device, rfp_items, rfp_ex)
            rm = RFP.passage_choice_logits(model, tok, device, rfp_items, rfp_ex)
            b["rfp_cloze"] = {"per_item": rc["per_item"], "hash": rc["per_item_hash"],
                              "acc_pct": rc["acc_pct"], "n": rc["n"]}
            b["rfp_mcq"] = {"per_item": rm["per_item"], "hash": rm["per_item_hash"],
                            "acc_pct": rm["acc_pct"], "n": rm["n"]}
            _acc.setdefault("rfp_cloze", {})[row] = rc["acc_pct"]
            _acc.setdefault("rfp_mcq", {})[row] = rm["acc_pct"]
            print(f"[LTX] {row} RFP cloze={rc['acc_pct']:.2f} mcq={rm['acc_pct']:.2f}", flush=True)
            _w()
        if "ppl" in axes:
            pp = TE.clean_text_perplexity(model, tok, device, max_tokens=a.ppl_max_tokens)
            b["ppl"] = {"ppl": pp["ppl"], "n_tokens": pp.get("n_tokens")}
            _acc.setdefault("wikitext_ppl", {})[row] = pp["ppl"]
            print(f"[LTX] {row} wikitext ppl={pp['ppl']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
        rep["elapsed_s"] = round(time.time() - t0, 1)
        _w()

    # ---- position-pairing tripwire: identical item order across every row ---------------
    for axis, key in (("mcq", "hash"), ("cloze", "hash"), ("rfp_cloze", "hash")):
        have = [r for r in rows if axis in banks[r]]
        if len(have) > 1:
            ref = banks[have[0]][axis][key]
            for r in have[1:]:
                assert banks[r][axis][key] == ref, f"{axis}: item order differs between {have[0]} and {r}"
    rep["pairing_verified"] = True
    _w()
    print("LACUNA_TEXT_OK", json.dumps(_acc, default=str)[:1500], flush=True)


if __name__ == "__main__":
    main()
