"""MERCATOR Wave 0 — the two GPU-light arms (4 L40S-h total).

  M-8  (queue 8)  The single-model sink predictor. Accumulate per-coordinate gradient magnitude
                  and sign-consistency on a FROZEN batch of the BASE model alone, score the
                  suppression statistic, and compare the predicted address with the MEASURED bank.
                  REFUTER: predictor <= activation-magnitude baseline on overlap with the banks.

  M-A6 (queue A6) DOME-nuisance vs sink overlap. DOME's object is the top eigenspace of the
                  CENTERED gradient covariance; restricted to one matrix's input coordinates that
                  is Sum_t (G_t - Gbar)^T (G_t - Gbar), whose top-k eigenvectors live on exactly
                  the axis every input-side bank lives on. Principal angles against core32 and
                  against the twin battery. REFUTER: overlap within the twin band.

Both arms share ONE gradient pass over the same frozen windows, which is why they are one job:
running them separately would double the GPU cost and — worse — would let the two arms see
different data, so any difference between them could be the batch rather than the object.

INDEPENDENCE (§7 circularity row). Every statistic here comes from the BASE model's own forward and
backward. No fine-tune, no consensus Gram and no bank enters the PREDICTION; the banks are used
only to SCORE it afterwards.

Reuses invariant_core_func's PG1 loading and blank-image perplexity verbatim (§11 reuse ledger):
those functions already carry the eager-attention requirement, the trailing-newline text-block
location, and the base_ppl gate that this arm's preflight needs.

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
from infra.pg import mercator_predict as PR
from infra.pg import mercator_program as P

ARMS = ("M-8", "M-A6")
FULL_BATTERY = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2")
BASE_PPL_TOL = 0.02          # relative; the anchor is recomputed, never transcribed


def _target_keys(model, types, layers):
    """The LANGUAGE-MODEL decoder weight names this pass differentiates.

    Scoped to the language model on purpose: PaliGemma carries a SigLIP vision tower whose blocks
    are ALSO named `layers.N.self_attn.q_proj.weight`, so a bare suffix match is ambiguous and
    resolves to two parameters — which is how the arm failed on its first run. The banks this
    prediction is scored against are the LM decoder's, so the vision tower is not merely
    redundant here, it is the wrong object.
    """
    from infra.pg.entities import resolve_submodule_paths
    lm_prefix = resolve_submodule_paths(model)["language_model"]
    names = dict(model.named_parameters())
    out = {}
    for t in types:
        for L in layers:
            suffix = f".layers.{L}.{'self_attn' if t[0] in 'qkvo' else 'mlp'}.{t}.weight"
            hits = [n for n in names if n.endswith(suffix) and n.startswith(lm_prefix)]
            assert len(hits) == 1, (
                f"{suffix} under {lm_prefix!r}: {len(hits)} matches (expected 1); "
                f"all suffix matches: {[n for n in names if n.endswith(suffix)]}")
            out[(t, L)] = hits[0]
    return out


def _windows(processor, n_seq: int, window: int = 512):
    """The FROZEN batch: the same wikitext windows every model in the roster sees."""
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


def _backward_pass(model, processor, device, texts, keys, blank, per_step):
    """One backward per window over the frozen texts; `per_step(key, grad_fp32)` sees each gradient.

    Factored out because the arm needs TWO passes over the SAME windows and they must be the same
    windows — a centered covariance computed against a mean from different data is not a centered
    covariance.
    """
    import torch
    params = dict(model.named_parameters())
    n_steps = 0
    for i, txt in enumerate(texts):
        inp = processor(text=[txt], images=[blank], return_tensors="pt", padding=False).to(device)
        model.zero_grad(set_to_none=True)
        out = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                    pixel_values=inp["pixel_values"].to(torch.bfloat16),
                    labels=inp["input_ids"])
        assert out.loss is not None, "no loss returned — labels were not accepted"
        out.loss.backward()
        for k in keys.values():
            g = params[k].grad
            assert g is not None, f"{k} received no gradient"
            per_step(k, g.to(torch.float32))
        n_steps += 1
        if (i + 1) % 5 == 0:
            print(f"[grad] {i + 1}/{len(texts)} windows", flush=True)
    model.zero_grad(set_to_none=True)
    return n_steps


def _activation_rms(model, processor, device, texts, keys, blank):
    """Per-input-coordinate activation RMS at each target module's own input, over the SAME frozen
    windows the gradients were taken on.

    Same windows on purpose: if the two addresses were measured on different data, a difference
    between the predictor and its named incumbent could be the batch rather than the statistic.
    The hook is registered on the module (not the parameter) because the incumbent's address is
    defined by what the matrix READS, and for o_proj that is the attention output rather than the
    residual stream.
    """
    import numpy as np
    import torch
    mods = dict(model.named_modules())
    mod_names = {k: k[: -len(".weight")] for k in keys.values()}
    for k, mn in mod_names.items():
        assert mn in mods, f"module {mn} not found for parameter {k}"
    acc = {k: None for k in mod_names}
    n = {k: 0 for k in mod_names}

    def hook(key):
        def f(_m, inputs, _o):
            x = inputs[0].detach().to(torch.float64)
            x = x.reshape(-1, x.shape[-1])
            s = x.pow(2).sum(dim=0)
            acc[key] = s if acc[key] is None else acc[key] + s
            n[key] += int(x.shape[0])
        return f

    handles = [mods[mn].register_forward_hook(hook(k)) for k, mn in mod_names.items()]
    with torch.no_grad():
        for txt in texts:
            inp = processor(text=[txt], images=[blank], return_tensors="pt",
                            padding=False).to(device)
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"].to(torch.bfloat16))
    for h in handles:
        h.remove()

    out = {}
    for k in mod_names:
        assert acc[k] is not None and n[k] > 0, f"{k} was never invoked — no activations captured"
        v = np.sqrt((acc[k] / n[k]).cpu().numpy())
        assert np.isfinite(v).all(), f"{k}: non-finite activation RMS"
        out[k] = v
    return out


def _grad_and_cov(model, processor, device, texts, keys, blank):
    """M-8's per-coordinate statistics and M-A6's centered covariance, over the same frozen batch.

    TWO passes, deliberately:

      pass 1  accumulates Sum_t G_t (which gives both the sign-consistency numerator and Gbar) and
              Sum_t ||G_t[:, j]||.
      pass 2  accumulates Sum_t (G_t - Gbar)^T (G_t - Gbar) directly.

    The one-pass alternative — accumulate Sum G^T G and subtract n Gbar^T Gbar at the end — is
    algebraically identical and numerically much worse: when gradients ARE consistent the two terms
    are nearly equal and the difference is catastrophic cancellation, which is exactly the regime
    where the sink's coordinates live. Centering BEFORE squaring costs one extra pass and has no
    cancellation, so the covariance can be accumulated in fp32 on the GPU. That matters: fp64
    matmul on an L40S runs at 1/64 rate, and the fp64 one-pass form would have taken hours.
    """
    import torch
    params = dict(model.named_parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for k in keys.values():
        params[k].requires_grad_(True)

    n_in = {k: int(params[k].shape[1]) for k in keys.values()}
    run_sum = {k: torch.zeros_like(params[k], dtype=torch.float32) for k in keys.values()}
    sum_norm = {k: torch.zeros(n_in[k], dtype=torch.float64, device=device) for k in keys.values()}

    def _p1(k, g32):
        run_sum[k] += g32
        sum_norm[k] += g32.to(torch.float64).pow(2).sum(dim=0).sqrt()

    n_steps = _backward_pass(model, processor, device, texts, keys, blank, _p1)
    assert n_steps >= 2, "the centered covariance needs at least two steps"
    mean = {k: (run_sum[k] / n_steps) for k in keys.values()}

    cov = {k: torch.zeros((n_in[k], n_in[k]), dtype=torch.float32, device=device)
           for k in keys.values()}

    def _p2(k, g32):
        c = g32 - mean[k]
        cov[k] += c.T @ c

    n2 = _backward_pass(model, processor, device, texts, keys, blank, _p2)
    assert n2 == n_steps, f"the two passes saw {n_steps} and {n2} windows"

    stats = {}
    for k in keys.values():
        s_col = run_sum[k].to(torch.float64).pow(2).sum(dim=0).sqrt().cpu().numpy()
        t_col = sum_norm[k].cpu().numpy()
        c = cov[k].to(torch.float64).cpu().numpy()
        c = 0.5 * (c + c.T)                 # symmetrise the fp32 accumulation before eigh
        cons, counts = PR.consistency(s_col, t_col, return_counts=True)
        stats[k] = {"mag": t_col / n_steps, "cons": cons, "cov": c, "n_steps": n_steps,
                    "zero_gradient": counts}
    return stats


def main():
    ap = argparse.ArgumentParser(description="MERCATOR Wave 0 — M-8 predictor + M-A6 DOME overlap")
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--n-seq", type=int, default=24, help="frozen windows in the batch")
    ap.add_argument("--n-ppl-seq", type=int, default=20, help="windows for the base_ppl gate")
    ap.add_argument("--k", type=int, default=H.BANK_K)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    types = [t.strip() for t in a.types.split(",") if t.strip()]
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    for t in types:
        assert t in H.TYPES, f"unknown type {t}"

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import _blank_image, _load_full_pg1, _ppl_blankimg

    assert torch.cuda.is_available(), "no CUDA — these are the GPU arms"
    device = "cuda"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, device)
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)

    gates = H.Gates(out)
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w0") / "expectations.jsonl")
    refs = {arm: exp.require(arm) for arm in ARMS}

    # PREFLIGHT: the base_ppl gate. The anchor is RECOMPUTED from the FUNC artifact in the §2
    # preflight and cited here by value, so this gate compares two measurements rather than a
    # measurement against a remembered number.
    ppl = _ppl_blankimg(model, processor, device, a.n_ppl_seq)
    ppl_val = float(ppl["ppl"] if isinstance(ppl, dict) else ppl)
    gates.record("base_ppl_measured", True, ppl=ppl_val, n_seq=a.n_ppl_seq,
                 note=("compare against the FUNC anchor recomputed in results/mercator_preflight3/"
                       "out/anchors/anchors.json; this run's window count differs from FUNC's, so "
                       "the two are not required to match to tolerance"))

    keys = _target_keys(model, types, layers)
    texts = _windows(processor, a.n_seq)
    blank = _blank_image(128)
    stats = _grad_and_cov(model, processor, device, texts, keys, blank)
    # The arm's NAMED incumbent is the activation-magnitude address, measured at each target
    # module's own INPUT — which differs by type (q/k/v read the post-norm residual, o_proj reads
    # the attention output), so it is hooked per module rather than shared.
    act = _activation_rms(model, processor, device, texts, keys, blank)

    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))

    # ---------------- M-8 ----------------
    meta8 = P.ARMS["M-8"]
    at8 = M.Atlas(out / "M-8", arm="M-8", run_id="m_8", register=meta8["register"],
                  wave=meta8["wave"], expected_ref=refs["M-8"], queue_item=meta8["queue_item"])
    m8 = {}
    for (t, L), key in sorted(keys.items()):
        st = stats[key]
        score = PR.suppression_score(st["mag"], st["cons"])
        U = WT.load_bank(bank_dir, t, L, "core32")
        cmp_ = PR.compare_to_baselines(score=score, act_rms=act[key], grad_mag=st["mag"],
                                       U_bank=U, k=a.k,
                                       seed=H.cell_seed(a.seed, "M-8", t, L))
        for name in ("predictor", "activation_magnitude", "gradient_magnitude", "random"):
            at8.add("address_overlap_jaccard", value=float(cmp_[name]["jaccard"]), layer=L,
                    type=t, bank_id="core32", model_id=WT.PG1_REPO, model_sha=base_sha,
                    incumbent_id=None if name == "predictor" else name,
                    flags={"scorer": name, "k": a.k, "n_steps": st["n_steps"],
                           "log_p_hypergeom": cmp_[name]["log_p_hypergeom"],
                           "enrichment": cmp_[name]["enrichment"],
                           "n_intersection": cmp_[name]["n_intersection"]})
        at8.add("gradient_sign_consistency_mean", value=float(np.mean(st["cons"])), layer=L,
                type=t, model_id=WT.PG1_REPO, model_sha=base_sha,
                flags={"n_steps": st["n_steps"], **st["zero_gradient"]})
        at8.add("zero_gradient_column_fraction", value=st["zero_gradient"]["zero_fraction"],
                layer=L, type=t, model_id=WT.PG1_REPO, model_sha=base_sha,
                flags={**st["zero_gradient"],
                       "why": "bf16 underflow makes some columns exactly zero at every step"})
        np.save(out / "M-8" / f"predicted_score__{t}__L{L}.npy", score)
        m8[f"{t}/L{L}"] = {k2: cmp_[k2] for k2 in
                           ("predictor", "activation_magnitude", "gradient_magnitude", "random",
                            "beats_activation_baseline", "beats_gradient_magnitude_baseline")}
    beats = sum(1 for v in m8.values() if v["beats_activation_baseline"])
    beats_g = sum(1 for v in m8.values() if v["beats_gradient_magnitude_baseline"])
    gates.record("m8_beats_activation_baseline_fraction", beats > len(m8) / 2,
                 n_cells=len(m8), n_beating=beats,
                 note="the REFUTER's named incumbent")
    gates.record("m8_beats_gradient_magnitude_baseline_fraction", beats_g > len(m8) / 2,
                 n_cells=len(m8), n_beating=beats_g,
                 note=("the harder control: does the sign-consistency factor add anything over "
                       "gradient magnitude alone?"))
    info8 = at8.to_parquet()

    # ---------------- M-A6 ----------------
    metaA6 = P.ARMS["M-A6"]
    atA = M.Atlas(out / "M-A6", arm="M-A6", run_id="m_a6", register=metaA6["register"],
                  wave=metaA6["wave"], expected_ref=refs["M-A6"], queue_item=metaA6["queue_item"])
    ma6 = {}
    for (t, L), key in sorted(keys.items()):
        cov = stats[key]["cov"]
        w, V = np.linalg.eigh(cov)
        dome = np.ascontiguousarray(V[:, -a.k:])          # top-k of the CENTERED covariance
        battery = WT.load_battery(bank_dir, t, L)
        battery["topsigma32"] = WT.load_bank(bank_dir, t, L, "topsigma32")
        cell = {}
        for bid, U in battery.items():
            ang = H.principal_angles(dome, U)
            atA.add("dome_principal_angle_mean", value=float(ang.mean()), layer=L, type=t,
                    bank_id=bid, model_id=WT.PG1_REPO, model_sha=base_sha,
                    flags={"k": a.k, "n_steps": stats[key]["n_steps"],
                           "object": "top-k eigenspace of the centered gradient covariance"})
            cell[bid] = float(ang.mean())
        haar = [cell[b] for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
        ma6[f"{t}/L{L}"] = {"angles": cell, "haar_mean": float(np.mean(haar)),
                            "core_minus_haar": cell["core32"] - float(np.mean(haar))}
        atA.add("dome_core_minus_haar_angle", value=ma6[f"{t}/L{L}"]["core_minus_haar"], layer=L,
                type=t, bank_id="core32", model_id=WT.PG1_REPO, model_sha=base_sha,
                flags={"haar_mean_angle": float(np.mean(haar))})
    atA.check_controls_per_statistic(["dome_principal_angle_mean"])
    infoA = atA.to_parquet()

    for arm, res, info in (("M-8", m8, info8), ("M-A6", ma6, infoA)):
        meta = P.ARMS[arm]
        M.write_json(out / arm, f"{arm}_result.json",
                     {"arm": arm, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                      "expected_ref": refs[arm], "provenance": M.provenance(),
                      "utc": M.utc_now(), "atlas": info, "base_repo": WT.PG1_REPO,
                      "base_sha": base_sha, "types": types, "layers": layers,
                      "n_frozen_windows": a.n_seq, "base_ppl_measured": ppl_val,
                      "independence": ("every prediction statistic comes from the BASE model's own "
                                       "forward/backward; banks are used only to score it"),
                      "result": res})
    print("W0GPU_OK " + json.dumps({"m8_rows": info8["n_rows"], "ma6_rows": infoA["n_rows"],
                                    "m8_cells_beating_baseline": beats, "n_cells": len(m8),
                                    "base_ppl": ppl_val}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
