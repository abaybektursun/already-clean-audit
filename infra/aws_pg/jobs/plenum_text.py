"""PLENUM Arm 1 — the TEXT battery over the whole roster (spec §2.1 text axes).

`lacuna_text` is REUSED for the four axes themselves (`text_eval.mmlu_choice_logits`,
`text_eval.mmlu_cloze`, `read_from_passage.passage_cloze` / `passage_choice_logits`,
`text_eval.clean_text_perplexity`) — this job changes only WHICH units they run on. LACUNA's job
hardcodes four ladder rows; the map needs `behavior/{text_recall,reading,text_ppl}` for all 32
units, which is 170 cells that were left PENDING because no job covered them.

Every unit enters through the same `plenum_units.install_unit` path the mm battery uses, so a text
row and an mm row for the same unit are the same object measured twice, not two harnesses.

Per-item scores are banked raw; labels and S are computed by `plenum_aggregate` from the bank.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.plenum_text --type g6e.2xlarge --no-weights \
      --args "--units pg1,matched_dw,ft-vqav2" --s3-prefix results/plenum_text_x --budget-min 700
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

PPL_ANCHOR = 22.42583
PPL_REL_TOL = 1e-3


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--axes", default="mcq,cloze,rfp,ppl")
    ap.add_argument("--mmlu-n-per-subject", type=int, default=0, help="0 = the full bank")
    ap.add_argument("--rfp-n", type=int, default=2000)
    ap.add_argument("--rfp-seed", type=int, default=20260620)
    ap.add_argument("--ppl-max-tokens", type=int, default=120_000)
    ap.add_argument("--stitch-seed", type=int, default=1)
    ap.add_argument("--matched-dw-seed", type=int, default=20260727)
    ap.add_argument("--rand-dec-seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    units = [u.strip() for u in a.units.split(",") if u.strip()]
    axes = [x.strip() for x in a.axes.split(",") if x.strip()]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_units as PU
    from infra.pg import read_from_passage as RFP
    from infra.pg import text_eval as TE

    assert torch.cuda.is_available(), "plenum_text needs GPU"
    device = "cuda"
    t0 = time.time()
    seeds = {"stitch": a.stitch_seed, "matched_dw": a.matched_dw_seed,
             "rand_dec": a.rand_dec_seed}
    n_per = a.mmlu_n_per_subject or None
    rep = {"object": "plenum_text_v1", "units": units, "axes": axes,
           "mmlu_n_per_subject": n_per, "rfp_n": a.rfp_n,
           "gpu": torch.cuda.get_device_name(0), "seeds": seeds}
    cells, banks = [], {}
    rep["cells"] = cells                       # bound BEFORE the loops (fleet lesson)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
        (out / "text_banks.json").write_text(json.dumps(banks, default=str))
    _w()

    rfp_items, rfp_ex = None, None
    if "rfp" in axes:
        built = RFP.build_synth(a.rfp_seed, a.rfp_n)
        rfp_items, rfp_ex = built[0], built[1]
        rep["rfp"] = {"n": len(rfp_items), "content_hash": RFP.content_hash(rfp_items)}
        _w()

    model, proc = LS.load_pg1(device=device)
    pg1_sd = LS.pg1_baseline_sd(model)
    tok = proc.tokenizer
    ppl0 = IP.ppl_gate(model, proc, device)
    val = float(ppl0["ppl"] if isinstance(ppl0, dict) else ppl0)
    rel = abs(val - PPL_ANCHOR) / PPL_ANCHOR
    rep["P2_ppl_anchor"] = {"ppl": val, "rel_diff": rel, "passes": bool(rel < PPL_REL_TOL)}
    assert rep["P2_ppl_anchor"]["passes"], f"P2 FAIL: {val}"
    _w()

    for uid in PU.unit_order(units):
        # gemma-2b is a DIFFERENT ARCHITECTURE from the PG1 shell, so `install_unit` cannot load it
        # and it was skipped entirely — leaving its text row PENDING. But the text axes need only a
        # causal LM and a tokenizer (they never touch the vision path), so the pre-install reference
        # is measurable here directly. Its image faculties remain structurally impossible, which the
        # map already records.
        if uid == "gemma-2b":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                LS.GEMMA1_REPO, torch_dtype=torch.bfloat16, device_map=device,
                attn_implementation="eager").eval()
            tok = AutoTokenizer.from_pretrained(LS.GEMMA1_REPO)
            info = {"unit": uid, "kind": "base_lm_native", "repo": LS.GEMMA1_REPO,
                    "note": "text axes only; no vision path exists for this unit"}
        else:
            info = PU.install_unit(uid, model, proc, pg1_sd, device, seeds)
        rep.setdefault("unit_info", {})[uid] = info
        b = {}
        banks[uid] = b
        print(f"[PTX] installed {uid}", flush=True)
        _w()
        if "mcq" in axes:
            mc = TE.mmlu_choice_logits(model, tok, device, n_per_subject=n_per)
            b["mcq"] = {"per_item": mc["per_item"], "hash": mc["per_item_hash"],
                        "acc_pct": mc["acc_pct"], "n": mc["n"]}
            cells.append({"unit": uid, "axis": "mmlu_mcq", "acc_pct": mc["acc_pct"], "n": mc["n"]})
            print(f"[PTX] {uid} MMLU-MCQ={mc['acc_pct']:.2f}", flush=True)
            _w()
        if "cloze" in axes:
            cz = TE.mmlu_cloze(model, tok, device, n_per_subject=n_per)
            b["cloze"] = {"logprobs": cz["cloze_logprobs"], "ntoks": cz["cloze_ntoks"],
                          "correct_idx": cz["correct_idx"], "hash": cz["per_item_hash"],
                          "acc_pct": cz["acc_pct"], "n": cz["n"]}
            cells.append({"unit": uid, "axis": "mmlu_cloze", "acc_pct": cz["acc_pct"], "n": cz["n"]})
            print(f"[PTX] {uid} MMLU-CLOZE={cz['acc_pct']:.2f}", flush=True)
            _w()
        if "rfp" in axes:
            rc = RFP.passage_cloze(model, tok, device, rfp_items, rfp_ex)
            rm = RFP.passage_choice_logits(model, tok, device, rfp_items, rfp_ex)
            b["rfp_cloze"] = {"per_item": rc["per_item"], "hash": rc["per_item_hash"],
                              "acc_pct": rc["acc_pct"], "n": rc["n"]}
            b["rfp_mcq"] = {"per_item": rm["per_item"], "hash": rm["per_item_hash"],
                            "acc_pct": rm["acc_pct"], "n": rm["n"]}
            cells.append({"unit": uid, "axis": "rfp_cloze", "acc_pct": rc["acc_pct"], "n": rc["n"]})
            cells.append({"unit": uid, "axis": "rfp_mcq", "acc_pct": rm["acc_pct"], "n": rm["n"]})
            print(f"[PTX] {uid} RFP cloze={rc['acc_pct']:.2f} mcq={rm['acc_pct']:.2f}", flush=True)
            _w()
        if "ppl" in axes:
            pp = TE.clean_text_perplexity(model, tok, device, max_tokens=a.ppl_max_tokens)
            b["ppl"] = {"ppl": pp["ppl"], "n_tokens": pp.get("n_tokens")}
            cells.append({"unit": uid, "axis": "wikitext_ppl", "ppl": pp["ppl"]})
            print(f"[PTX] {uid} wikitext ppl={pp['ppl']:.4f}", flush=True)
            _w()

    rep["elapsed_s"] = round(time.time() - t0, 1)
    _w()
    print(f"PLENUM_TEXT_OK units={len(units)} cells={len(cells)}", flush=True)


if __name__ == "__main__":
    main()
