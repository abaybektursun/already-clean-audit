"""M-12 — the sink x induction-heads bridge (queue item 12). Wave 1, runs FIRST (§8).

Hypothesis-to-kill: induction heads are principal READERS of the sink. If true it would jointly
explain the q-side concentration, the criticality, the ICL affinity, and fine-tuning's forced
avoidance — which is why this arm runs before anything else in the wave.

Stages (`--stage`):

  identify   (a) Find PG1's induction heads by the standard repeated-random-token method, then
                 measure each head's READ energy in the sink versus the full twin battery.
                 REFUTER: induction heads' read overlap with the sink is within the twin band.
  probe      (c) Resolve conserved-core's WAIVED position-vs-induction discriminator: the two
                 synthetic corpora (induction-pure repeat-copy, position-pure explicit retrieval),
                 scored under core ablation against the twin battery, each with the label-shuffle
                 positive control that CAN fail.

The two-clock overlay (b) needs the OLMo checkpoint suite and shares its loader with M-A4; it is
`--stage two_clock` and is run once those checkpoints are pulled.

WHY THE READ ENERGY IS THE RIGHT STATISTIC. A head reads the residual stream through its own q (and
k) slice, so "how much of this head's read lives in subspace U" is the energy of that slice inside
U. For a Haar k-subspace of R^n the expectation is k/n, which is the band the REFUTER is written
against — the null here is the twin battery, not zero.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_induction as I
from infra.pg import mercator_program as P

ARM = "M-12"
STAGES = ("identify", "probe", "two_clock")
FULL_BATTERY = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2")


def _attn_on_repeat(model, processor, device, seq_len: int, n_seq: int, seed: int):
    """Per-(layer, head) prefix-matching scores, averaged over `n_seq` repeated-random sequences.

    The random tokens are drawn from the tokenizer's vocabulary excluding every special id, so no
    head can score by attending to a delimiter. PG1 needs an image, so a blank gray one is supplied
    and the text block is located exactly (the processor appends a trailing newline, which is why
    the offset is found rather than assumed).
    """
    import torch
    from infra.aws_pg.jobs.invariant_core_func import _blank_image, _locate_text_block

    tok = processor.tokenizer
    specials = set(int(x) for x in tok.all_special_ids)
    toks = I.repeated_random_tokens(int(tok.vocab_size), seq_len, n_seq, seed, forbid=specials)
    blank = _blank_image(128)

    acc = None
    for i in range(n_seq):
        ids = torch.tensor(toks[i], dtype=torch.long)
        text = tok.decode(ids.tolist())
        # Re-tokenising the decoded string can change the id sequence, so the block we score is the
        # one the processor actually produced — located by exact match on the re-tokenised ids.
        win = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        inp = processor(text=[text], images=[blank], return_tensors="pt", padding=False).to(device)
        with torch.inference_mode():
            o = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                      pixel_values=inp["pixel_values"].to(torch.bfloat16),
                      output_attentions=True)
        pos = _locate_text_block(inp["input_ids"][0], win)
        # The decoded/retokenised block must still be a clean [R, R] of even length, or the
        # induction target offset is wrong. Skipping the check would silently score noise.
        n_text = int(win.shape[0])
        if n_text % 2 != 0:
            continue
        L = n_text // 2
        att = np.stack([a[0].to(torch.float32).cpu().numpy() for a in o.attentions])  # (Lyr,H,T,T)
        per_layer = np.stack([I.prefix_matching_score(att[l], L, offset=pos)
                              for l in range(att.shape[0])])                          # (Lyr, H)
        acc = per_layer if acc is None else acc + per_layer
        del o, att
    assert acc is not None, "no usable repeated sequence survived retokenisation"
    return acc / n_seq, toks.shape


def run_identify(model, processor, device, at, gates, a, base_sha) -> dict:
    scores, tok_shape = _attn_on_repeat(model, processor, device, a.seq_len, a.n_seq, a.seed)
    sel = I.induction_heads(scores, threshold=I.INDUCTION_THRESHOLD)
    n_layers, n_heads = scores.shape
    for L in range(n_layers):
        for h in range(n_heads):
            at.add("prefix_matching_score", value=float(scores[L, h]), layer=L, type="q_proj",
                   item_id=f"head{h}", model_id=WT.PG1_REPO, model_sha=base_sha,
                   flags={"head": h, "threshold": I.INDUCTION_THRESHOLD,
                          "is_induction_head": bool(scores[L, h] >= I.INDUCTION_THRESHOLD),
                          "method": "repeated-random-token prefix matching",
                          "seq_len": a.seq_len, "n_seq": a.n_seq})
    gates.record("m12_any_induction_head_found", sel["n_heads_selected"] > 0, **{
        k: sel[k] for k in ("threshold", "n_heads_selected", "n_heads_total", "score_mean",
                            "score_max", "score_p95")})

    # Per-head READ energy in each bank. The head's q slice is its read of the residual stream.
    bank_dir = M.fetch_all_banks(H.cache_dir(pathlib.Path(a.out), "banks"))
    import torch
    lm = model.get_submodule(_lm_prefix(model))
    cells, head_dim = {}, None
    for L in range(n_layers):
        Wq = _layer_weight(lm, L, "q_proj").to(torch.float64).cpu().numpy()
        head_dim = Wq.shape[0] // n_heads
        battery = {b: WT.load_bank(bank_dir, "q_proj", L, b) for b in FULL_BATTERY}
        for h in range(n_heads):
            e = I.head_battery(Wq, battery, head=h, head_dim=head_dim)
            for b, v in e.items():
                at.add("head_read_energy", value=float(v), layer=L, type="q_proj", bank_id=b,
                       item_id=f"head{h}", model_id=WT.PG1_REPO, model_sha=base_sha,
                       flags={"head": h, "head_dim": head_dim,
                              "is_induction_head": bool(scores[L, h] >= I.INDUCTION_THRESHOLD),
                              "flat_expectation": H.BANK_K / Wq.shape[1]})
            cells[f"L{L}h{h}"] = {"prefix_matching": float(scores[L, h]),
                                  "is_induction_head": bool(scores[L, h] >= I.INDUCTION_THRESHOLD),
                                  **e}
    at.check_controls_per_statistic(["head_read_energy"])

    # THE REFUTER, evaluated: do the induction heads read the sink more than the twin band?
    ind = [c for c in cells.values() if c["is_induction_head"]]
    non = [c for c in cells.values() if not c["is_induction_head"]]
    verdict = None
    if ind:
        core = np.array([c["core32"] for c in ind])
        twins = np.array([[c[b] for b in ("haar32_s0", "haar32_s1", "haar32_s2")] for c in ind])
        band_lo, band_hi = float(twins.min()), float(twins.max())
        s = M.surprise(float(core.mean()), float(twins.mean()), twins.ravel().tolist())
        verdict = {"n_induction_heads": len(ind), "n_other_heads": len(non),
                   "core32_mean": float(core.mean()), "twin_mean": float(twins.mean()),
                   "twin_band": [band_lo, band_hi],
                   "core_over_twin": float(core.mean() / twins.mean()),
                   "within_twin_band": bool(band_lo <= core.mean() <= band_hi),
                   "surprise": s}
        at.add("induction_head_core_read_energy", value=float(core.mean()), bank_id="core32",
               type="q_proj", model_id=WT.PG1_REPO, model_sha=base_sha, n=len(ind),
               flags={"twin_mean": float(twins.mean()), "twin_band": [band_lo, band_hi],
                      "refuter": "within the twin band kills the bridge"})
    return {"stage": "identify", "selection": sel, "scores_shape": list(scores.shape),
            "token_shape": list(tok_shape), "head_dim": head_dim,
            "refuter_cell": verdict, "cells": cells}


def _lm_prefix(model) -> str:
    from infra.pg.entities import resolve_submodule_paths
    return resolve_submodule_paths(model)["language_model"]


def _layer_weight(lm, layer: int, t: str):
    from infra.pg.gemma_scope import resolve_gemma2_layers
    layers = resolve_gemma2_layers(lm)
    parent = layers[layer].self_attn if t[0] in "qkvo" else layers[layer].mlp
    return getattr(parent, t).weight.detach()


def run_probe(model, processor, device, at, gates, a, base_sha) -> dict:
    """(c) The position-vs-induction discriminator, with a control that can fail."""
    from infra.pg import conserved_core_probes as CP
    corpora = I.discriminator_corpora(a.n_items)
    out = {}
    for name, items in corpora.items():
        golds = [g for _p, g in items]
        shuffled = CP.shuffle_labels(np.arange(len(golds)), a.seed)
        out[name] = {"n_items": len(items),
                     "example_prompt": items[0][0][:160], "example_gold": items[0][1],
                     "shuffle_control_is_a_permutation":
                         bool(sorted(np.asarray(shuffled).tolist()) == list(range(len(golds))))}
        at.add("discriminator_corpus_size", value=float(len(items)), corpus_id=name,
               flags={"source": "conserved_core_probes (reused unforked)",
                      "pure_for": name})
    gates.require("m12_probe_corpora_built",
                  all(v["n_items"] == a.n_items for v in out.values()), **{
                      k: v["n_items"] for k, v in out.items()})
    gates.require("m12_shuffle_control_is_a_permutation",
                  all(v["shuffle_control_is_a_permutation"] for v in out.values()),
                  per_corpus={k: bool(v["shuffle_control_is_a_permutation"])
                              for k, v in out.items()},
                  n_corpora=len(out),
                  note="a control that is not a permutation of the labels cannot fail honestly")
    return {"stage": "probe", "corpora": out,
            "note": ("the corpora and their shuffle control are built and verified here; the "
                     "ablation contrast that consumes them is the arm's next cell and needs the "
                     "core-ablation forward, which is scheduled with the M-G reader work")}


def main():
    ap = argparse.ArgumentParser(description="M-12 sink x induction heads")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="identify", choices=list(STAGES))
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--n-seq", type=int, default=8)
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id=f"m_12_{a.stage}", register=meta["register"],
                 wave=meta["wave"], expected_ref=expected_ref, queue_item=meta["queue_item"])
    gates = H.Gates(out)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import _load_full_pg1

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)

    if a.stage == "identify":
        res = run_identify(model, processor, "cuda", at, gates, a, base_sha)
    elif a.stage == "probe":
        res = run_probe(model, processor, "cuda", at, gates, a, base_sha)
    else:
        raise AssertionError("two_clock runs with the OLMo checkpoint loader (shared with M-A4)")

    info = at.to_parquet()
    M.write_json(out, f"{ARM}_{a.stage}_result.json",
                 {"arm": ARM, "stage": a.stage, "queue_item": meta["queue_item"],
                  "refuter": meta["refuter"], "expected_ref": expected_ref,
                  "provenance": M.provenance(), "utc": M.utc_now(), "atlas": info,
                  "base_repo": WT.PG1_REPO, "base_sha": base_sha, "result": res})
    print(f"M12_{a.stage}_OK " + json.dumps({"rows": info["n_rows"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
