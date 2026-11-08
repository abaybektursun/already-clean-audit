"""MENDEL MN-7 — the SGD span signature, on deltas this job produces (queue item 15b).

THE CLAIM. SGD's weight update is a sum of outer products,  dW = -eta * sum_t delta_t a_t^T,  so
its ROW SPACE lies inside the span of the inputs the layer actually saw. Adam's is not: the
m_hat/sqrt(v_hat) normalisation rescales coordinates independently, so the update leaves that span.
The banked figure the program carries is "Adam ~36% outside".

WHY THIS RUNS INSTEAD OF READING A BANKED CHECKPOINT. The arm's blocker was that no banked SGD
delta exists — `tenant_x4`/`copernicus_m4c` bank the CORE-PROJECTED trajectory, not dW — and the
recorded unlock is "one short SGD run keeping the delta". Reading a banked ADAM fine-tune against
a banked SGD one would in any case have confounded the optimizer with the corpus, the steps and
the learning rate, since the community fine-tunes share none of those. This job runs BOTH
optimizers from the same initial weights, on the same windows in the same order, for the same
number of steps, so the optimizer is the only thing that differs.

WHY THE STATISTIC IS ENERGY-WEIGHTED, not a rank test. After a few hundred steps the layer has
seen ~10^5 token rows in a 2048-dimensional space, so the raw activation span is the whole space
and "is dW in the span" is trivially yes for any optimizer. The question with content is WHERE in
that space: how much of dW lies in the top-r activation eigenspace, on a ladder of r. G4b banks
exactly that basis (top 512 per site, median 94% of the activation energy) and its own metrics
name MN-7 as the consumer.

    in_span(r)  = ||dW V_r||_F / ||dW||_F          (ENDOW's write_fraction, reused verbatim)
    out_span(r) = sqrt(1 - in_span(r)^2)

MATCHED-RANDOM NULL. A dW with a haar row space has in_span(r) = sqrt(r/d) in expectation, so the
null is not "zero" — it is a band, and it is computed per cell rather than assumed. The statistic
is a ratio of norms and so is SCALE-FREE: the two optimizers moving different distances cannot by
itself move it. ||dW||_F is reported per site anyway.

FROZEN RULE, at the primary r = 512:
  SPAN-CLEAN  iff SGD's out-of-span is below Adam's (paired CI clears zero) AND below the
              matched-random band, on cells spanning >= 2 types and >= 2 layers
  OUT-OF-SPAN iff SGD's out-of-span is NOT below the matched-random band — SGD's motion is as
              out-of-span as a random one, which kills the signature
  AMBIGUOUS   otherwise

PRECISION. The model is loaded in FLOAT32, not the bf16 every other MENDEL job uses. bf16 carries
8 mantissa bits, so a small update is rounded onto a coarse grid and much of it to zero — which
would corrupt the DIRECTION of dW, the only thing this arm measures. The span basis was harvested
in bf16 by G4b; that affects the basis by rounding at ~1e-3 relative and the top eigenvectors are
stable to it, while quantising dW itself would not be survivable.

GPU. FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP
from infra.pg.endow_writefrac import write_fraction

ARM = "MN-7"
TYPES = H.TYPES_RESID
G4B_PREFIX = f"{MC.RESULTS_S3}/mendel_g4b_v1/out"
R_LADDER = (8, 32, 128, 512)
R_PRIMARY = 512
N_HAAR = 16
TRAINABLE = ("q_proj", "k_proj", "v_proj", "o_proj")


def _train_windows(processor, n: int, window: int = 512):
    """Windows from wikitext-2 TRAIN, built exactly like G4/G4b's frozen TEST batch.

    The split differs on purpose. G4b harvested the activation basis on TEST; training on those
    same windows and then measuring against a basis estimated from them would make the span
    self-fulfilling. Training on TRAIN asks the harder question — does the signature hold against
    an independently estimated basis.
    """
    from datasets import load_dataset
    tok = processor.tokenizer
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    out = []
    for start in range(0, ids.size(0), window):
        chunk = ids[start:start + window]
        if chunk.size(0) < 8:
            break
        out.append(tok.decode(chunk.tolist()))
        if len(out) >= n:
            break
    assert len(out) == n, f"only {len(out)} train windows available, needed {n}"
    return out


def trainable_sites(model):
    """{(type, layer): module} for the projections this job trains and measures."""
    from infra.pg.entities import resolve_submodule_paths
    lm = resolve_submodule_paths(model)["language_model"]
    mods = dict(model.named_modules())
    out = {}
    for t in TRAINABLE:
        for L in range(H.N_LAYERS):
            name = f"{lm}.layers.{L}.self_attn.{t}"
            assert name in mods, f"no module {name} — the decoder path moved"
            out[(t, L)] = name
    return out


def run_optimizer(model, sites, texts, processor, blank, *, opt_name: str, lr: float,
                  steps: int, init: dict, device: str) -> dict:
    """Train from `init` with one optimizer and return dW per site.

    The initial weights are restored from `init` rather than the model being reloaded, so both
    optimizers provably start from the SAME numbers rather than from two loads that ought to agree.
    """
    mods = dict(model.named_modules())
    for key, name in sites.items():
        mods[name].weight.data.copy_(init[key])
    params = [mods[name].weight for name in sites.values()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = (torch.optim.SGD(params, lr=lr, momentum=0.0) if opt_name == "sgd"
           else torch.optim.Adam(params, lr=lr))
    img_id = model.config.image_token_index

    model.train()
    losses = []
    for i in range(steps):
        inp = processor(text=[texts[i]], images=[blank], return_tensors="pt",
                        padding=False).to(device)
        labels = inp["input_ids"].clone()
        labels[labels == img_id] = -100          # the image prefix is not a prediction target
        out = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                    pixel_values=inp["pixel_values"].to(model.dtype), labels=labels)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss))
        if (i + 1) % 25 == 0:
            print(f"[mn7] {opt_name} {i + 1}/{steps} loss {np.mean(losses[-25:]):.4f}", flush=True)
    model.eval()
    return ({key: (mods[name].weight.data - init[key]).to(torch.float64).cpu().numpy()
             for key, name in sites.items()},
            {"first_25_loss": float(np.mean(losses[:25])),
             "last_25_loss": float(np.mean(losses[-25:])), "n_steps": steps, "lr": lr})


def span_row(dW, V, r: int, seed: int) -> dict:
    """in/out-of-span at rank r, beside a matched-random band of the same rank."""
    Vr = np.ascontiguousarray(V[:, :r])
    ins = float(write_fraction(dW, Vr))
    rng = np.random.default_rng(seed)
    band = [float(write_fraction(dW, np.linalg.qr(rng.standard_normal((V.shape[0], r)))[0]))
            for _ in range(N_HAAR)]
    oos = float(np.sqrt(max(0.0, 1.0 - ins ** 2)))
    band_oos = [float(np.sqrt(max(0.0, 1.0 - b ** 2))) for b in band]
    return {"r": r, "in_span": ins, "out_of_span": oos,
            "random_in_span_mean": float(np.mean(band)),
            "random_out_of_span_mean": float(np.mean(band_oos)),
            "random_out_of_span_lo": float(np.min(band_oos)),
            "random_out_of_span_hi": float(np.max(band_oos)),
            "analytic_random_in_span": float(np.sqrt(r / V.shape[0])),
            "below_random_band": bool(oos < np.min(band_oos))}


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-7 — the SGD span signature")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr-sgd", type=float, default=1e-3)
    ap.add_argument("--lr-adam", type=float, default=1e-5)
    a = ap.parse_args()

    # FIRST, before any transformers/datasets import. run_job.sh exports HF_HUB_OFFLINE=1, under
    # which HfApi().model_info raises OfflineModeIsEnabled and load_dataset cannot reach the hub.
    # I reasoned past this because G4b makes the same calls and ran — but G4b ran against a
    # different huggingface_hub build, and "a sibling job did it" is not evidence about this one.
    # It cost MN-7 a launch.
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()

    from huggingface_hub import HfApi
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import _blank_image

    assert torch.cuda.is_available(), "no CUDA — MN-7 trains"
    device = "cuda"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn7", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn7"), out)

    base_sha = HfApi().model_info(WT.PG1_REPO).sha
    g4b = json.loads(man.require("g4b_metrics", f"{G4B_PREFIX}/metrics.json").read_text())
    assert g4b["base_sha"] == base_sha, (
        f"G4b banked the span for {g4b['base_sha']} but this run resolved {base_sha} — the span "
        f"and the delta must describe the same model")
    assert g4b["top_k"] >= R_PRIMARY, f"G4b banked only {g4b['top_k']} directions"
    vec = man.require("g4b_vectors", f"{G4B_PREFIX}/vectors", recursive=True)

    # FLOAT32: see the precision note in the module docstring.
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        WT.PG1_REPO, revision=base_sha, torch_dtype=torch.float32,
        device_map=device, attn_implementation="eager")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    blank = _blank_image(128)
    sites = trainable_sites(model)
    texts = _train_windows(processor, a.steps)
    mods = dict(model.named_modules())
    init = {k: mods[n].weight.data.clone() for k, n in sites.items()}
    print(f"[mn7] {len(sites)} trainable sites, {a.steps} windows, fp32", flush=True)

    deltas, runs = {}, {}
    for opt_name, lr in (("sgd", a.lr_sgd), ("adam", a.lr_adam)):
        deltas[opt_name], runs[opt_name] = run_optimizer(
            model, sites, texts, processor, blank, opt_name=opt_name, lr=lr,
            steps=a.steps, init=init, device=device)
        print(f"[mn7] {opt_name}: {runs[opt_name]}", flush=True)

    # ---- both optimizers must actually have MOVED, or the contrast is between two zeros --------
    norms = {o: {f"{t}_L{L}": float(np.linalg.norm(d)) for (t, L), d in deltas[o].items()}
             for o, _ in (("sgd", 0), ("adam", 0))}
    for o in norms:
        m = min(norms[o].values())
        gates.record(f"mn7_{o}_actually_moved", m > 0.0, min_delta_norm=m,
                     median_delta_norm=float(np.median(list(norms[o].values()))))
        assert m > 0.0, f"{o} left some site exactly unchanged — nothing to measure"

    cells = []
    for (t, L) in sorted(sites):
        # Through the shared loader: G4b banks the eigenVECTORS truncated to 512 and the
        # eigenVALUES in full, so re-sorting by the full spectrum and indexing the truncated
        # matrix is an IndexError. It killed this job once, after the weight sync and 400
        # training steps, on the first site it touched.
        V, _ea, _share = MC.load_activation_eigenbank(vec, t, L)
        for r in R_LADDER:
            for o in ("sgd", "adam"):
                row = span_row(deltas[o][(t, L)], V, r, H.cell_seed(a.seed, ARM, o, t, L, r))
                cells.append({"optimizer": o, "type": t, "layer": L,
                              "delta_norm": norms[o][f"{t}_L{L}"], **row})
                if r == R_PRIMARY:
                    # SGD is the treatment; ADAM IS THE STRUCTURED CONTROL. It is a real
                    # optimizer trained on identical data for identical steps, which is exactly
                    # what §5 means by structured — not a random subspace. Filing both as
                    # `treatment` left the arm with no structured member and
                    # check_control_trinity would have asserted at the very end, after all
                    # training. Caught by reproducing the assert offline, not by a launch.
                    at.add("out_of_span_fraction", value=row["out_of_span"], layer=L, type=t,
                           bank_id=o,
                           control_id=MC.TREATMENT if o == "sgd" else MC.STRUCTURED,
                           flags={"r": r, "in_span": row["in_span"],
                                  "delta_norm": norms[o][f"{t}_L{L}"]})
                    at.add("out_of_span_fraction", value=row["random_out_of_span_mean"], layer=L,
                           type=t, bank_id=o, control_id=MC.HAAR,
                           flags={"r": r, "n_haar": N_HAAR,
                                  "analytic": row["analytic_random_in_span"]})
        print(f"[mn7] {t} L{L}: sgd oos@{R_PRIMARY}="
              f"{[c['out_of_span'] for c in cells if c['optimizer'] == 'sgd' and c['type'] == t and c['layer'] == L and c['r'] == R_PRIMARY][0]:.4f} "
              f"adam="
              f"{[c['out_of_span'] for c in cells if c['optimizer'] == 'adam' and c['type'] == t and c['layer'] == L and c['r'] == R_PRIMARY][0]:.4f}",
              flush=True)

    # ---- the structured control: the SAME dW with its rows permuted --------------------------
    # A shuffled dW keeps every entry and every marginal and destroys only the alignment between
    # the update and the activation basis. It is the shuffle member of the trinity, and it is a
    # sharper null than haar because it also preserves dW's spectrum.
    rng = np.random.default_rng(a.seed)
    shuf = []
    for (t, L) in sorted(sites):
        V = MC.load_activation_eigenbank(vec, t, L)[0][:, :R_PRIMARY]
        d = deltas["sgd"][(t, L)]
        p = rng.permutation(d.shape[1])
        shuf.append(float(np.sqrt(max(0.0, 1.0 - float(write_fraction(d[:, p], V)) ** 2))))
    at.add_ci("out_of_span_fraction", shuf, bank_id="sgd", control_id=MC.SHUFFLE, seed=a.seed,
              flags={"what": "SGD dW with input coordinates permuted — same entries, same "
                             "spectrum, alignment to the activation basis destroyed"})

    prim = {o: [c for c in cells if c["optimizer"] == o and c["r"] == R_PRIMARY]
            for o in ("sgd", "adam")}
    paired = MC.paired_delta_ci([c["out_of_span"] for c in prim["sgd"]],
                                [c["out_of_span"] for c in prim["adam"]], seed=a.seed)
    shuf_paired = MC.paired_delta_ci([c["out_of_span"] for c in prim["sgd"]], shuf, seed=a.seed)
    below = [c for c in prim["sgd"] if c["below_random_band"]]
    n_types = len({c["type"] for c in below})
    n_layers = len({c["layer"] for c in below})
    sgd_below_adam = bool(paired["ci_hi"] < 0)
    if sgd_below_adam and n_types >= 2 and n_layers >= 2:
        verdict = "SPAN-CLEAN"
    elif not below:
        verdict = "OUT-OF-SPAN"
    else:
        verdict = "AMBIGUOUS"
    verdict = MP.check_verdict(ARM, verdict)

    gates.record("mn7_sgd_below_adam_out_of_span", sgd_below_adam, **paired)
    gates.record("mn7_sgd_below_its_own_shuffle", bool(shuf_paired["ci_hi"] < 0), **shuf_paired)
    gates.record("mn7_sgd_below_the_matched_random_band", len(below) > 0,
                 n_cells_below=len(below), n_cells=len(prim["sgd"]),
                 n_types=n_types, n_layers=n_layers)
    at.check_control_trinity(["out_of_span_fraction"])
    info = at.to_parquet()
    man.write()

    by_r = {r: {o: float(np.median([c["out_of_span"] for c in cells
                                    if c["optimizer"] == o and c["r"] == r]))
                for o in ("sgd", "adam")} for r in R_LADDER}
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"], "base_sha": base_sha,
        "trainable_types": list(TRAINABLE), "n_sites": len(sites),
        "r_primary": R_PRIMARY, "r_ladder": list(R_LADDER), "runs": runs,
        "sgd_minus_adam_out_of_span": paired,
        "sgd_minus_its_own_shuffle": shuf_paired,
        "median_out_of_span_by_r": by_r,
        "n_cells_below_random_band": len(below), "below_types": n_types,
        "below_layers": n_layers,
        "delta_norms": norms,
        "scope_note": (
            "q/k/v/o_proj across all 18 decoder layers are trained and measured; gate/up/down_proj "
            "are frozen. Training all seven would need ~1.8 B trainable parameters and their Adam "
            "states, which does not fit beside an fp32 3 B model on one L40S. The restriction is "
            "IDENTICAL for both optimizers, which is what the contrast requires; it narrows the "
            "claim to attention projections and is stated rather than glossed."),
        "cells": cells, "atlas": info})
    print("MN7_OK " + json.dumps({
        "verdict": verdict, "sgd_minus_adam": round(paired["delta_mean"], 5),
        "ci": [round(paired["ci_lo"], 5), round(paired["ci_hi"], 5)],
        "median_by_r": {str(k): {o: round(v, 4) for o, v in d.items()} for k, d in by_r.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
