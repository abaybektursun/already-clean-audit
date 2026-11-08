"""M-7 — quantization continuation: does the protection list transfer, and why does it compose
badly? (queue item 7). Wave P, Register B.

    (a) REPLICATION ON FOREIGN FAMILIES. Quantizing a weight matrix to `bits` and keeping a small
        PROTECTION LIST of input columns at full precision is the standard recipe; the question is
        which columns. The house answer is the sink's; the field's answer is the outlier columns
        (largest activation magnitude). This arm runs both on Llama and Qwen, from THEIR OWN
        re-census banks, at MATCHED protection budget.
        REFUTER: the protection list is no better than outlier-column selection on the foreign
        families.

    (b) THE COMPOSITION FAILURE. Protecting list A recovers x, protecting list B recovers y, and
        protecting both recovers less than x + y. The mechanism this arm proposes is trivial once
        stated and has never been measured: the two lists SHARE coordinates, so the second list's
        marginal contribution is only its non-shared part. The arm measures the overlap and asks
        whether it accounts for the shortfall.

MATCHED BUDGET IS THE WHOLE COMPARISON. Both lists protect the same NUMBER of columns, so the
comparison is about which columns and never about how many, and the protected bytes are identical
by construction.

THE QUANTIZER IS WRITTEN OUT — per-column absmax to `bits`, the simplest thing that is honest —
because a library quantizer would bring its own column selection and the arm would be measuring
that instead.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-7"
BASES = ("meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B")
# A CONSTANT HERE GOES STALE BY CONSTRUCTION — see the long note in `mercator_foreign.py`. The
# launcher refuses a prefix that already has a STATUS, so a retried dependency always lands under a
# new name, and an empty `s3 sync` succeeds silently. Overridden with `--banks-s3` (M-D99).
RECENSUS_S3 = "results/mercator_recensus/out/banks"
BITS = 4
SPIKE_MULT = 5.0
LISTS = ("none", "sink", "outlier", "random", "sink+outlier")


def quantize_absmax(W, bits: int, protect=None):
    """Per-column absmax quantization to `bits`, with `protect` columns left at full precision.

    Per COLUMN because a column is one residual-stream input coordinate, which is the axis both
    candidate protection lists are expressed on; per-row or per-tensor scaling would make the two
    lists incomparable.
    """
    import torch
    Wq = W.to(torch.float32).clone()
    n_levels = 2 ** (bits - 1) - 1
    scale = Wq.abs().amax(dim=0, keepdim=True).clamp(min=1e-12) / n_levels
    Q = torch.round(Wq / scale).clamp(-n_levels - 1, n_levels) * scale
    if protect is not None and len(protect) > 0:
        idx = torch.as_tensor(np.asarray(protect), device=W.device, dtype=torch.long)
        Q[:, idx] = Wq[:, idx]
    return Q.to(W.dtype)


def main():
    ap = argparse.ArgumentParser(description="M-7 — quantization continuation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bases", default=",".join(BASES))
    ap.add_argument("--banks-s3", default=RECENSUS_S3,
                    help="comma-list of re-census bank prefixes, synced in the order given")
    ap.add_argument("--bits", type=int, default=BITS)
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--n-calib", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bases = [b.strip() for b in a.bases.split(",") if b.strip()]
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_wp") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_7", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import gc
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from infra.aws_pg.jobs.mercator_recensus import bank_name

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    bank_dir = H.cache_dir(out, "recensus_banks")
    bank_dir.mkdir(parents=True, exist_ok=True)
    bank_index, bank_origin, complete_bases = M.load_recensus_banks(a.banks_s3, bank_dir)
    # A census killed by the reaper streams half-written banks to S3 that look finished on the file
    # listing; `recensus_partial.json` is the completeness signal, so an unfinished base is dropped.
    skipped_incomplete = [b for b in bases if b not in complete_bases]
    bases = [b for b in bases if b in complete_bases]
    if skipped_incomplete:
        print(f"[M-7] SKIP (census did not complete): {skipped_incomplete}", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())

    def ppl(model, tok):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        tot, n = 0.0, 0
        for i in range(min(a.n_seq, ids.shape[0] // a.window)):
            b = ids[i * a.window:(i + 1) * a.window].unsqueeze(0).to("cuda")
            with torch.inference_mode():
                lg = model(input_ids=b).logits[:, :-1].to(torch.float32)
            tot += float(torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1), reduction="sum"))
            n += int(b[:, 1:].numel())
        return float(np.exp(tot / n))

    results = {}
    for base in bases:
        cfg = AutoConfig.from_pretrained(base)
        n_layers = int(getattr(cfg, "num_hidden_layers", 0))
        sink_by_layer = {}
        for L in range(n_layers):
            p = bank_index.get(bank_name(base, "q_proj", L, "core32"))
            if p is not None:
                U = np.load(p)
                lev = H.leverage(U)
                sink_by_layer[L] = np.flatnonzero(
                    lev > SPIKE_MULT * (U.shape[1] / U.shape[0]))
        if not sink_by_layer:
            results[base] = {"outcome": M.COVERAGE_IMPOSSIBLE,
                             "why": "no re-census core32 banks for this base"}
            at.add("quant_banks_available", value=0.0, model_id=base,
                   flags={**results[base], "outcome": M.COVERAGE_IMPOSSIBLE})
            print(f"[M-7] {base}: no banks — coverage-impossible", flush=True)
            continue
        sha = api.model_info(base).sha
        tok = AutoTokenizer.from_pretrained(base, revision=sha)
        model = AutoModelForCausalLM.from_pretrained(
            base, revision=sha, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="eager").eval()
        layers = model.model.layers
        mods = {L: layers[L].self_attn.q_proj for L in sink_by_layer}
        pristine = {L: m.weight.detach().clone() for L, m in mods.items()}
        base_ppl = ppl(model, tok)

        # THE OUTLIER LIST, measured on this model with this calibration text — the field's
        # selection rule, computed here rather than cited, so both lists come from the same run.
        buf = {}

        def mk(L):
            def hook(_m, args):
                h = args[0].detach().to(torch.float32)
                buf[L] = h.reshape(-1, h.shape[-1]).abs().mean(0)
            return hook

        hs = [mods[L].register_forward_pre_hook(mk(L)) for L in mods]
        ids = tok(text[:20000], return_tensors="pt",
                  add_special_tokens=False).input_ids[:, :a.window].to("cuda")
        with torch.inference_mode():
            model(input_ids=ids)
        for h in hs:
            h.remove()
        rng = np.random.default_rng(H.cell_seed(a.seed, ARM, base, "random_list"))
        lists = {}
        overlap = {}
        for L in mods:
            k = int(len(sink_by_layer[L]))
            out_idx = torch.topk(buf[L], k).indices.cpu().numpy()
            n_cols = int(mods[L].weight.shape[1])
            rnd = rng.choice(n_cols, size=k, replace=False)
            both = np.unique(np.concatenate([sink_by_layer[L], out_idx]))
            lists[L] = {"sink": sink_by_layer[L], "outlier": out_idx, "random": rnd,
                        "sink+outlier": both, "none": np.array([], dtype=int)}
            inter = len(set(sink_by_layer[L].tolist()) & set(out_idx.tolist()))
            overlap[L] = {"k": k, "n_shared": inter,
                          "jaccard": inter / max(len(both), 1),
                          "union_size": int(len(both))}

        cell = {"base_ppl": base_ppl, "n_layers_protected": len(mods),
                "protected_columns_per_layer": {str(L): int(len(sink_by_layer[L]))
                                                for L in mods},
                "overlap": {str(L): overlap[L] for L in mods}, "bits": a.bits}
        for name in LISTS:
            with torch.no_grad():
                for L, m in mods.items():
                    m.weight.data.copy_(quantize_absmax(pristine[L], a.bits,
                                                        protect=lists[L][name]))
            p_ = ppl(model, tok)
            cell[name] = {"ppl": p_, "damage": p_ / base_ppl,
                          "n_protected": int(sum(len(lists[L][name]) for L in mods))}
            at.add("quant_ppl", value=p_, bank_id=("core32" if name == "sink" else None),
                   model_id=base, model_sha=sha, corpus_id=f"wikitext2:window{a.window}",
                   flags={"protection_list": name, "bits": a.bits, "base_ppl": base_ppl,
                          "damage": p_ / base_ppl, **cell[name]})
            print(f"[M-7] {base.split('/')[-1]:16s} protect={name:14s} "
                  f"n={cell[name]['n_protected']:5d} ppl={p_:9.4f} "
                  f"damage={p_ / base_ppl:.4f}", flush=True)
            with torch.no_grad():
                for L, m in mods.items():
                    m.weight.data.copy_(pristine[L])

        # (b) THE COMPOSITION ARITHMETIC. Recovery is measured against the unprotected quantization.
        d_none = cell["none"]["ppl"] - base_ppl
        rec = {n: (d_none - (cell[n]["ppl"] - base_ppl)) / d_none if d_none > 0 else float("nan")
               for n in ("sink", "outlier", "sink+outlier")}
        additive = rec["sink"] + rec["outlier"]
        shortfall = additive - rec["sink+outlier"]
        mean_jac = float(np.mean([overlap[L]["jaccard"] for L in mods]))
        cell["recovery"] = rec
        cell["composition"] = {"additive_prediction": additive,
                               "measured": rec["sink+outlier"], "shortfall": shortfall,
                               "mean_list_jaccard": mean_jac,
                               "shortfall_over_overlap": (shortfall / mean_jac
                                                          if mean_jac > 0 else float("nan"))}
        at.add("quant_composition_shortfall", value=float(shortfall), model_id=base,
               model_sha=sha,
               flags={**cell["composition"],
                      "definition": ("(recovery_sink + recovery_outlier) - "
                                     "recovery_both; positive means composing the two "
                                     "protections buys less than their sum")})
        print(f"[M-7] {base.split('/')[-1]:16s} recovery sink={rec['sink']:.4f} "
              f"outlier={rec['outlier']:.4f} both={rec['sink+outlier']:.4f} "
              f"shortfall={shortfall:+.4f} list-overlap={mean_jac:.4f}", flush=True)
        results[base] = cell
        del model
        gc.collect()
        torch.cuda.empty_cache()
        M.s3_sync_incremental(out)

    got = {k: v for k, v in results.items() if v.get("outcome") is None}
    gates.record("m7_any_foreign_base_had_banks", bool(got), bases=sorted(got),
                 outcome=None if got else M.COVERAGE_IMPOSSIBLE)
    if got:
        beats = [b for b, v in got.items() if v["sink"]["ppl"] < v["outlier"]["ppl"]]
        gates.record("m7_protection_list_beats_outlier_selection", bool(beats),
                     beating_bases=beats,
                     sink_ppl={b: v["sink"]["ppl"] for b, v in got.items()},
                     outlier_ppl={b: v["outlier"]["ppl"] for b, v in got.items()},
                     random_ppl={b: v["random"]["ppl"] for b, v in got.items()},
                     note=("matched budget: both lists protect the same number of columns, so "
                           "the comparison is which columns and never how many"))
        gates.record("m7_composition_shortfall_is_positive",
                     bool(all(v["composition"]["shortfall"] > 0 for v in got.values())),
                     shortfall={b: v["composition"]["shortfall"] for b, v in got.items()},
                     list_overlap={b: v["composition"]["mean_list_jaccard"]
                                   for b, v in got.items()},
                     note=("the mechanism under test: the two lists share coordinates, so the "
                           "second buys only its non-shared part"))
        b0 = next(iter(got))
        sur.add(arm=ARM, expected_ref=expected_ref, cell=b0, statistic="quant_ppl/core32",
                s=M.surprise(got[b0]["sink"]["ppl"], got[b0]["outlier"]["ppl"],
                             [got[b0]["outlier"]["ppl"], got[b0]["random"]["ppl"]]))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "bases": bases, "bits": a.bits, "lists": list(LISTS),
                  "banks_source": a.banks_s3, "n_bank_files": len(bank_index),
                  "banks_by_prefix": dict(bank_origin),
                  "complete_census_bases": sorted(complete_bases),
                  "skipped_incomplete_census": skipped_incomplete,
                  "quantizer": "per-column absmax, written out in this job", "results": results})
    print("M7_OK " + json.dumps({"rows": info["n_rows"], "bases": sorted(got)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
