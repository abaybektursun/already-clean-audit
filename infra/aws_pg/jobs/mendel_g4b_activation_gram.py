"""MENDEL G4b — the activation SPAN, which nothing in this program had banked.

WHY THIS EXISTS. Every activation artifact in the program stores only a DIAGONAL: G4's `act_rms`
is sqrt(mean x^2) per coordinate, SHACKLETON's anchor `actstats.npz` stores `sum_sq` per
coordinate. A diagonal cannot give a span — the off-diagonal is exactly the part that says which
DIRECTIONS the activations occupy. Two arms need the span and neither could run without it:

  MN-7  "SGD's dW provably stays inside the activation span; Adam puts ~36% outside." The test is
        the fraction of dW's row energy inside span{x}. Undefined from a diagonal.
  MN-2 / MN-13  their energy-matched control is currently built in the COORDINATE basis from
        per-coordinate energy, which is a legitimate but coordinate-aligned control. With the real
        eigenbasis the control becomes the one `conserved_core_probes.energy_matched_subspace` was
        written for.

WHAT IT BANKS. Per (type, layer) site on the 6 residual types: the top-K eigenvectors of the
activation second-moment matrix E[x x^T] and its FULL eigenvalue spectrum. Not the Gram itself —
2048^2 fp32 x 108 sites is 1.8 GB of mostly-redundant bytes, while the top-K basis plus the full
spectrum is what both consumers actually read and is ~10x smaller.

`down_proj` is excluded: its input axis is 16384, so its Gram is 2.1 GB per site and 38 GB over the
layers. Disclosed, matching freeze A3's exclusion of the same type elsewhere.

THE SAME FROZEN BATCH as G4, byte-identical window construction, so the span and the per-coordinate
statistics describe the same data and can be used in one expression.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP

ARM = "G4"                    # G4b is G4's second half; it shares the gate's roster row
TYPES = H.TYPES_RESID         # 6 residual types, n = 2048; down_proj excluded (see docstring)
TOP_K = 512
G4_PREFIX = f"{MC.RESULTS_S3}/mendel_g4_v1/out"


def _windows(processor, n_seq: int, window: int = 512):
    """THE frozen batch — byte-identical to G4's and to mercator_w0_gpu's."""
    from datasets import load_dataset
    tok = processor.tokenizer
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    out = []
    for start in range(0, ids.size(0), window):
        chunk = ids[start:start + window]
        if chunk.size(0) < 8:
            break
        out.append(tok.decode(chunk.tolist()))
        if len(out) >= n_seq:
            break
    assert len(out) == n_seq, f"only {len(out)} windows available, needed {n_seq}"
    return out


def _site_modules(model, types, layers):
    from infra.pg.entities import resolve_submodule_paths
    lm = resolve_submodule_paths(model)["language_model"]
    mods = dict(model.named_modules())
    out = {}
    for t in types:
        for L in layers:
            suffix = f".layers.{L}.{'self_attn' if t[0] in 'qkvo' else 'mlp'}.{t}"
            hits = [n for n in mods if n.endswith(suffix) and n.startswith(lm)]
            assert len(hits) == 1, f"{suffix} under {lm!r}: {len(hits)} matches"
            out[(t, L)] = hits[0]
    return out


def main():
    ap = argparse.ArgumentParser(description="MENDEL G4b — activation Gram eigenbasis")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    (out / "vectors").mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_g4b", wave=meta["wave"], queue_item=meta["item"])
    man = MC.Manifest(H.cache_dir(out, "mendel_g4b"), out)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import _blank_image, _load_full_pg1

    assert torch.cuda.is_available(), "no CUDA — G4b runs a forward pass"
    device = "cuda"
    base_sha = HfApi().model_info(WT.PG1_REPO).sha
    g4 = json.loads(man.require("g4_metrics", f"{G4_PREFIX}/metrics.json").read_text())
    assert g4["base_sha"] == base_sha, (
        f"G4 ran on {g4['base_sha']} but this run resolved {base_sha} — the span and the "
        f"per-coordinate statistics must describe the same model")
    assert g4["n_frozen_windows"] == a.n_seq, (
        f"G4 used {g4['n_frozen_windows']} windows, this run {a.n_seq} — the two halves must "
        f"describe the same data")

    model = _load_full_pg1(base_sha, device)
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    layers = list(range(H.N_LAYERS))
    sites = _site_modules(model, TYPES, layers)
    texts = _windows(processor, a.n_seq)
    blank = _blank_image(128)
    print(f"[g4b] {len(sites)} sites, {len(texts)} frozen windows", flush=True)

    # fp32 accumulation on the GPU: 2048^2 x 4 B x 108 sites = 1.8 GB, comfortable on an L40S, and
    # the matrix is a sum of PSD outer products so fp32 has no cancellation to suffer.
    acc = {k: torch.zeros((H.IN_DIM[k[0]],) * 2, dtype=torch.float32, device=device)
           for k in sites}
    ntok = {k: 0 for k in sites}
    mods = dict(model.named_modules())

    def hook(key):
        def f(_m, inputs, _o):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.float32)
            acc[key] += x.T @ x
            ntok[key] += int(x.shape[0])
        return f

    handles = [mods[name].register_forward_hook(hook(k)) for k, name in sites.items()]
    with torch.no_grad():
        for i, txt in enumerate(texts):
            inp = processor(text=[txt], images=[blank], return_tensors="pt",
                            padding=False).to(device)
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"].to(torch.bfloat16))
            if (i + 1) % 5 == 0:
                print(f"[g4b] {i + 1}/{len(texts)} windows", flush=True)
    for h in handles:
        h.remove()

    rows = []
    for (t, L), name in sorted(sites.items()):
        assert ntok[(t, L)] > 0, f"{t}/L{L} was never invoked"
        G = (acc[(t, L)] / ntok[(t, L)]).to(torch.float64)
        G = 0.5 * (G + G.T)                      # symmetrise the fp32 accumulation before eigh
        w, V = torch.linalg.eigh(G)              # ascending
        w = w.flip(0).cpu().numpy()              # DESCENDING, as energy_matched_subspace expects
        V = V.flip(1)[:, : a.top_k].cpu().numpy().astype(np.float32)
        assert np.isfinite(w).all() and np.isfinite(V).all(), f"{t}/L{L}: non-finite eigensystem"
        w = np.maximum(w, 0.0)                   # PSD up to fp error; a tiny negative is not a mode
        np.save(out / "vectors" / f"act_gram_evecs__{t}__L{L}.npy", V)
        np.save(out / "vectors" / f"act_gram_evals__{t}__L{L}.npy", w)
        share = float(w[: a.top_k].sum() / w.sum())
        pr = float(H.participation_ratio(w))
        rows.append({"type": t, "layer": L, "n_tokens": ntok[(t, L)],
                     "top_k_energy_share": share, "spectrum_participation_ratio": pr,
                     "lambda_max": float(w[0]), "lambda_min": float(w[-1])})
        at.add("activation_gram_topk_energy_share", value=share, layer=L, type=t,
               model_id=WT.PG1_REPO, model_sha=base_sha, n=ntok[(t, L)],
               flags={"top_k": a.top_k, "participation_ratio": pr,
                      "why": "how much of the activation energy the banked basis captures"})
        at.add("activation_gram_participation_ratio", value=pr, layer=L, type=t,
               model_id=WT.PG1_REPO, model_sha=base_sha,
               flags={"n_eigenvalues": int(w.size)})
        del acc[(t, L)]

    info = at.to_parquet()
    man.write()
    shares = [r["top_k_energy_share"] for r in rows]
    MC.write_metrics(out, {
        "arm": "G4b", "verdict": "BANKED", "queue_item": meta["item"],
        "base_repo": WT.PG1_REPO, "base_sha": base_sha, "n_frozen_windows": a.n_seq,
        "types": list(TYPES), "excluded_types": ["down_proj"],
        "exclusion_reason": "input axis 16384 -> 2.1 GB per site, 38 GB over layers (freeze A3)",
        "top_k": a.top_k, "n_sites": len(rows),
        "top_k_energy_share": {"min": float(np.min(shares)), "median": float(np.median(shares)),
                               "max": float(np.max(shares))},
        "vectors_banked": ["act_gram_evecs", "act_gram_evals"],
        "consumers": {"MN-7": "the activation span for the SGD out-of-span test",
                      "MN-2/MN-13": "a non-diagonal energy-matched control subspace"},
        "sites": rows, "atlas": info})
    print("G4B_OK " + json.dumps({"n_sites": len(rows), "top_k": a.top_k,
                                  "median_energy_share": round(float(np.median(shares)), 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
