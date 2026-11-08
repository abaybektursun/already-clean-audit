"""M-E3 — KV-cache quantization, and its long-context extension (queue E3 + item 7c). Wave 1.

Register B. §8 puts this before M-7, whose quantization continuation rides on it.

THE QUESTION. The house has shown that protecting the CORE columns of a weight matrix beats
protecting the outlier columns at 3-5 bits. The KV cache is a different object: it stores k and v
in HEAD space, not the residual stream, so the sink — a residual subspace — is not literally in it.
What IS in it is the IMAGE of the sink under the projection: k = W_k x, so the sink's contribution
to the cache lives in span(W_k U). The arm asks whether protecting THAT beats protecting the
plain top-magnitude dimensions, which is the standard KV-quantization move and the named incumbent.

MATCHED BITS IS THE WHOLE COMPARISON. Holding j of d dimensions at fp16 and the rest at b' bits
costs j*16 + (d-j)*b'; the uniform arm costs d*b. The arm solves for the b' that equalises them, so
the two arms differ ONLY in WHICH dimensions are protected — never in how many bits are spent. An
unmatched comparison here would be measuring the budget, not the routing.

THE READOUT IS PERPLEXITY, not reconstruction error. A cache quantization that reconstructs well
but degrades the model is not a win, and the reverse is what the arm is looking for. The cache is
quantized in place through a forward hook on k_proj/v_proj outputs, which IS the cache content.

Long-context extension (7c): the same contrast at an increasing context ladder, because the claim
is that activation outliers bite harder as the cache grows.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import darwin_ceq as CEQ
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-E3"
ARMS = ("uniform", "routed_sink_image", "routed_top_magnitude", "fp16")
INCUMBENT = "routed_top_magnitude"          # the standard KV-quant move


def matched_bits(d: int, j: int, bits_uniform: int) -> dict:
    """The bulk bit-width that makes `j` fp16-protected dims cost the same as uniform.

    j*16 + (d-j)*b' = d*b  =>  b' = (d*b - 16j) / (d - j)

    Returns the exact solution and the integer used, plus the realized totals, because b' must be a
    whole number of bits and the residual difference is part of the comparison rather than hidden.
    """
    assert 0 <= j < d, f"j={j} out of range for d={d}"
    exact = (d * bits_uniform - 16 * j) / (d - j)
    b = int(np.floor(exact))
    assert b >= 2, (
        f"protecting {j} of {d} dims at fp16 leaves {exact:.2f} bits for the bulk, below the "
        f"2-bit floor — the routed arm cannot be matched to {bits_uniform} bits at this j")
    tot_u = int(d * bits_uniform)
    tot_r = int(j * 16 + (d - j) * b)
    return {"bits_uniform": int(bits_uniform), "n_protected": int(j), "d": int(d),
            "bulk_bits_exact": float(exact), "bulk_bits": b,
            "total_bits_uniform": tot_u, "total_bits_routed": tot_r,
            "budget_slack_bits": tot_u - tot_r,
            "routed_budget_fraction": tot_r / tot_u,
            # THE ROUNDING IS DELIBERATELY AGAINST THE ROUTED ARMS. b' is floored, so the routed
            # configuration spends NO MORE than the uniform one and usually less. A routed win is
            # therefore a win at a smaller budget, and a routed loss has the slack as one of its
            # explanations — which is the direction that makes the comparison safe to read.
            "handicap": ("routed arms spend %.1f%% of the uniform budget; the floor is applied "
                         "against them, never for them" % (100.0 * tot_r / tot_u))}


def sink_image_dims(W_k, U, j: int) -> list:
    """The j cache dimensions most aligned with the SINK'S IMAGE under this projection.

    k = W_k x, so the sink's contribution to the cache is span(W_k U). Each cache dimension's
    alignment with it is the row norm of W_k U — how much of that output coordinate is driven by
    the suppressed input subspace.
    """
    A = np.asarray(W_k, dtype=np.float64) @ np.asarray(U, dtype=np.float64)
    score = (A ** 2).sum(axis=1)
    return sorted(int(i) for i in np.argsort(score)[::-1][:j])


def top_magnitude_dims(rms, j: int) -> list:
    """The incumbent: the j cache dimensions with the largest activation magnitude."""
    r = np.asarray(rms, dtype=np.float64)
    return sorted(int(i) for i in np.argsort(r)[::-1][:j])


class _QuantCache:
    """Quantize k_proj / v_proj OUTPUTS in place — that is exactly what the cache stores.

    `dims_by_key` maps each hooked module name to the dimensions held at fp16; everything else is
    quantized to `bits`. `arm='fp16'` leaves the forward untouched and is the ceiling.
    """

    def __init__(self, modules: dict, bits: int, dims_by_key: dict, enabled: bool = True):
        self.modules = modules
        self.bits = int(bits)
        self.dims = dims_by_key
        self.enabled = enabled
        self._h = []

    def __enter__(self):
        import torch
        if not self.enabled:
            return self

        def mk(key):
            def hook(_m, _inp, out):
                x = out.detach().to(torch.float64).cpu().numpy()
                q = CEQ.quantize_activations(x, self.bits, keep_dims=self.dims.get(key, ()))
                return torch.from_numpy(q).to(out.dtype).to(out.device)
            return hook

        for key, mod in self.modules.items():
            self._h.append(mod.register_forward_hook(mk(key)))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False


def main():
    ap = argparse.ArgumentParser(description="M-E3 KV-cache quantization at matched bits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--protect-frac", type=float, default=0.05)
    ap.add_argument("--contexts", default="128,256,512")
    ap.add_argument("--n-seq", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    contexts = [int(x) for x in a.contexts.split(",") if x.strip()]
    M.check_dose_grid(contexts, arm=ARM, min_doses=3, log_spaced=True)
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_e3", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    gates = H.Gates(out)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import (_blank_image, _decoder_layers,
                                                       _load_full_pg1, _ppl_blankimg)

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    dec = _decoder_layers(model)
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))

    kv_mods, dims_sink, dims_top = {}, {}, {}
    d_kv = None
    for L, layer in enumerate(dec):
        for t in ("k_proj", "v_proj"):
            mod = getattr(layer.self_attn, t)
            key = f"L{L}.{t}"
            kv_mods[key] = mod
            Wk = mod.weight.detach().to(torch.float64).cpu().numpy()
            d_kv = Wk.shape[0] if d_kv is None else d_kv
            assert Wk.shape[0] == d_kv, f"{key}: cache width {Wk.shape[0]} != {d_kv}"
            U = WT.load_bank(bank_dir, t, L, "core32")
            j = max(1, int(round(a.protect_frac * d_kv)))
            dims_sink[key] = sink_image_dims(Wk, U, j)
            # the incumbent's dims come from the WEIGHT's own row magnitude, which is the
            # weight-side stand-in for activation magnitude and needs no extra forward pass
            dims_top[key] = top_magnitude_dims((Wk ** 2).sum(axis=1), j)
    j = len(next(iter(dims_sink.values())))
    mb = matched_bits(d_kv, j, a.bits)
    overlap = np.mean([len(set(dims_sink[k]) & set(dims_top[k])) / max(j, 1) for k in dims_sink])
    at.add("kv_protected_dim_overlap", value=float(overlap),
           flags={**mb, "meaning": "how much the sink-image and top-magnitude dim sets agree"})
    gates.record("e3_arms_are_distinguishable", overlap < 0.9, overlap=float(overlap),
                 note="if the two dim sets coincide the arms are the same experiment")
    gates.require("e3_bits_matched", mb["budget_slack_bits"] >= 0, **mb)

    results = {}
    for ctx in contexts:
        row = {}
        for arm in ARMS:
            dims = {"uniform": {}, "routed_sink_image": dims_sink,
                    "routed_top_magnitude": dims_top, "fp16": {}}[arm]
            bits = a.bits if arm == "uniform" else mb["bulk_bits"]
            with _QuantCache(kv_mods, bits, dims, enabled=(arm != "fp16")):
                r = _ppl_blankimg(model, processor, "cuda", a.n_seq, window=ctx)
            row[arm] = float(r["ppl"])
            at.add("kv_quant_ppl", value=row[arm], dose=float(ctx),
                   incumbent_id=None if arm == "routed_sink_image" else arm,
                   corpus_id=f"wikitext2:window{ctx}", model_id=WT.PG1_REPO, model_sha=base_sha,
                   flags={"arm": arm, "bits": bits, "context": ctx, "n_seq": a.n_seq, **mb})
            print(f"[M-E3] ctx={ctx} {arm}: ppl={row[arm]:.4f}", flush=True)
        row["sink_minus_incumbent"] = row["routed_sink_image"] - row[INCUMBENT]
        row["uniform_minus_incumbent"] = row["uniform"] - row[INCUMBENT]
        row["degradation_vs_fp16"] = {k: row[k] - row["fp16"] for k in ARMS if k != "fp16"}
        results[ctx] = row

    at.check_readout_corpus(["kv_quant_ppl"])
    deltas = [results[c]["sink_minus_incumbent"] for c in contexts]
    gates.record("e3_sink_routing_beats_incumbent_at_every_context",
                 all(d < 0 for d in deltas), deltas=deltas, contexts=contexts,
                 note="lower ppl is better; the REFUTER is no better than uniform at matched bits")

    info = at.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "register": meta["register"],
                  "refuter": meta["refuter"], "expected_ref": expected_ref,
                  "provenance": M.provenance(), "utc": M.utc_now(), "atlas": info,
                  "base_repo": WT.PG1_REPO, "base_sha": base_sha, "incumbent": INCUMBENT,
                  "matched_bits": mb, "protected_dim_overlap": float(overlap),
                  "contexts": contexts, "results": results,
                  "readout_note": ("perplexity with the cache quantized in place, not "
                                   "reconstruction error: a quantization that reconstructs well "
                                   "and degrades the model is not a win")})
    print("E3_OK " + json.dumps({"rows": info["n_rows"], "deltas": deltas}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
