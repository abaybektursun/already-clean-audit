"""MENDEL G4 — the preflight (queue item 12), and MN-1/MN-4's shared input pass.

Two jobs in one GPU pass, because they need the SAME frozen batch and running them apart would
let the two see different data:

  G4   Reproduce M-8's banked address-prediction table THROUGH THE MN-1 INSTRUMENT PATH before any
       new number is produced. The banked target is `results/mercator_w0gpu4/out/M-8/
       M-8_result.json` — mean Jaccard 0.4160 for the ACTIVATION-MAGNITUDE address (the freeze
       record's amendment A6: the spec's "0.416-class" names the INCUMBENT, not the predictor,
       whose mean is 0.3068 and whose refuter fired on 60/72 cells).

  BANK The per-coordinate vectors the downstream arms need and `mercator_w0gpu4` did not keep
       (amendment A8): activation RMS at each target module's own input, gradient magnitude, and
       gradient sign-consistency, for ALL SEVEN types x 18 layers. MN-1 consumes the activation
       RMS as its traffic null; MN-4 consumes mag/cons as the raw material of the T* map.

The scoring is done by `mercator_predict.compare_to_baselines` REUSED VERBATIM and by
`mendel_transport.transport_cell`'s own address helpers, so "the MN-1 instrument path reproduces
the M-8 number" is a statement about the code MN-1 will actually run, not about a re-implementation
that happens to agree.

INDEPENDENCE. Every statistic comes from PG1's own forward/backward on a frozen wikitext batch.
The banks enter only to SCORE the prediction, never to make it.

Deliberately NOT recomputed here: M-A6's centered gradient covariance. It cost the original run
two backward passes and a 2048x2048 fp32 accumulation per cell for a statistic no MENDEL arm
consumes. One pass is enough for mag/cons, which halves the GPU time.

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
from infra.pg import mercator_predict as PR

ARM = "G4"
# The banked run this preflight is written against. Named as a POINTER; every number is refetched
# and recomputed here, never transcribed (§0 provenance rule).
M8_REFERENCE = f"{MC.RESULTS_S3}/mercator_w0gpu4/out/M-8/M-8_result.json"
# The four types the banked M-8 run covered. The preflight can only be compared on these; the
# other three are banked for the downstream arms but have no M-8 reference to reproduce.
M8_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj")
# Reproduction tolerance on the mean Jaccard. The comparison is against a run on the same base
# sha, the same frozen corpus and the same window count, so the only sources of drift are library
# version and GPU nondeterminism in the bf16 forward; 0.02 is ~5% of the target and would not hide
# a wrong address list, which is what this gate is actually guarding.
REPRO_TOL = 0.02


def _windows(processor, n_seq: int, window: int = 512):
    """THE frozen batch. Byte-identical to `mercator_w0_gpu._windows` — same dataset, same split,
    same joining rule, same window size, same truncation. A preflight that reproduced a number on
    a different batch would prove nothing about the instrument."""
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


def _target_keys(model, types, layers):
    """The LANGUAGE-MODEL decoder weight names. Scoped to the LM prefix on purpose: PaliGemma's
    SigLIP tower also has `layers.N.self_attn.q_proj.weight`, so a bare suffix match resolves to
    two parameters — the defect that failed M-8's first run (mercator_w0_gpu docstring)."""
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


def _grad_stats(model, processor, device, texts, keys, blank):
    """One backward per frozen window; per-INPUT-coordinate gradient magnitude and sign
    consistency.

    `cons_j = ‖Σ_t G_t[:,j]‖ / Σ_t ‖G_t[:,j]‖` needs the running SUM of gradients, which is not
    recoverable from per-step column norms — hence one fp32 matrix per target key held live.
    """
    import torch
    params = dict(model.named_parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for k in keys.values():
        params[k].requires_grad_(True)

    run_sum = {k: torch.zeros_like(params[k], dtype=torch.float32) for k in keys.values()}
    sum_norm = {k: torch.zeros(int(params[k].shape[1]), dtype=torch.float64, device=device)
                for k in keys.values()}
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
            g32 = g.to(torch.float32)
            run_sum[k] += g32
            sum_norm[k] += g32.to(torch.float64).pow(2).sum(dim=0).sqrt()
        n_steps += 1
        if (i + 1) % 5 == 0:
            print(f"[grad] {i + 1}/{len(texts)} windows", flush=True)
    model.zero_grad(set_to_none=True)
    assert n_steps >= 2, "sign consistency is undefined on a single step"

    stats = {}
    for k in keys.values():
        s_col = run_sum[k].to(torch.float64).pow(2).sum(dim=0).sqrt().cpu().numpy()
        t_col = sum_norm[k].cpu().numpy()
        cons, counts = PR.consistency(s_col, t_col, return_counts=True)
        stats[k] = {"mag": t_col / n_steps, "cons": cons, "n_steps": n_steps,
                    "zero_gradient": counts}
        assert np.isfinite(stats[k]["mag"]).all(), f"{k}: non-finite gradient magnitude"
    return stats


def _activation_rms(model, processor, device, texts, keys, blank):
    """Per-input-coordinate activation RMS at each target module's OWN input, over the SAME frozen
    windows the gradients were taken on.

    Hooked on the module rather than the parameter because the incumbent address is defined by what
    the matrix READS, and for o_proj / down_proj that is not the residual stream. Same windows on
    purpose: a difference between the predictor and its incumbent measured on different data could
    be the batch rather than the statistic.
    """
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


def main():
    ap = argparse.ArgumentParser(description="MENDEL G4 preflight + MN-1/MN-4 input bank")
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default=",".join(H.TYPES))
    ap.add_argument("--layers", default="")
    ap.add_argument("--n-seq", type=int, default=24, help="frozen windows — M-8 used 24")
    ap.add_argument("--k", type=int, default=H.BANK_K)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    (out / "vectors").mkdir(parents=True, exist_ok=True)
    types = [t.strip() for t in a.types.split(",") if t.strip()]
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    for t in types:
        assert t in H.TYPES, f"unknown type {t}"

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import _blank_image, _load_full_pg1

    assert torch.cuda.is_available(), "no CUDA — G4 runs a forward and a backward"
    device = "cuda"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, device)
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)

    man = MC.Manifest(H.cache_dir(out, "mendel_g4"), out)
    ref_path = man.require("m8_reference", M8_REFERENCE)
    m8_ref = json.loads(ref_path.read_text())
    assert m8_ref["base_sha"] == base_sha, (
        f"the banked M-8 run used base sha {m8_ref['base_sha']} but this run resolved {base_sha} — "
        f"a reproduction across base revisions is not a reproduction")

    bank_dir = WT.fetch_banks(H.cache_dir(out, "banks"))
    keys = _target_keys(model, types, layers)
    texts = _windows(processor, a.n_seq)
    blank = _blank_image(128)
    print(f"[g4] {len(keys)} cells, {len(texts)} frozen windows", flush=True)

    stats = _grad_stats(model, processor, device, texts, keys, blank)
    act = _activation_rms(model, processor, device, texts, keys, blank)

    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_g4", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)

    per_cell = {}
    for (t, L), key in sorted(keys.items()):
        st = stats[key]
        score = PR.suppression_score(st["mag"], st["cons"])
        U = WT.load_bank(bank_dir, t, L, "core32")
        cmp_ = PR.compare_to_baselines(score=score, act_rms=act[key], grad_mag=st["mag"],
                                       U_bank=U, k=a.k,
                                       seed=H.cell_seed(a.seed, "G4", t, L))
        per_cell[f"{t}/L{L}"] = {n: cmp_[n] for n in
                                 ("predictor", "activation_magnitude", "gradient_magnitude",
                                  "random", "beats_activation_baseline")}
        # The vectors the downstream arms consume. Banked BEFORE any scoring decision so a failed
        # gate still leaves MN-1/MN-4 their inputs (the incremental-save rule).
        np.save(out / "vectors" / f"act_rms__{t}__L{L}.npy", act[key])
        np.save(out / "vectors" / f"grad_mag__{t}__L{L}.npy", st["mag"])
        np.save(out / "vectors" / f"grad_cons__{t}__L{L}.npy", st["cons"])

        for name, cid in (("predictor", MC.TREATMENT), ("activation_magnitude", MC.STRUCTURED),
                          ("gradient_magnitude", MC.STRUCTURED), ("random", MC.HAAR)):
            at.add("address_overlap_jaccard", value=float(cmp_[name]["jaccard"]), layer=L, type=t,
                   bank_id="core32", model_id=WT.PG1_REPO, model_sha=base_sha, control_id=cid,
                   flags={"scorer": name, "k": a.k, "n_steps": st["n_steps"],
                          "n_intersection": cmp_[name]["n_intersection"],
                          "log_p_hypergeom": cmp_[name]["log_p_hypergeom"]})
        at.add("zero_gradient_column_fraction", value=st["zero_gradient"]["zero_fraction"],
               layer=L, type=t, model_id=WT.PG1_REPO, model_sha=base_sha,
               flags={**st["zero_gradient"],
                      "why": "bf16 underflow makes some columns exactly zero at every step"})

    # ---- the reproduction gate, on the four types the banked run covered ----
    ref_cells = m8_ref["result"]
    shared = sorted(set(per_cell) & set(ref_cells))
    assert shared, f"no overlapping cells with the banked run (ours: {sorted(per_cell)[:3]})"
    repro = {}
    for scorer in ("activation_magnitude", "predictor", "gradient_magnitude", "random"):
        ours = float(np.mean([per_cell[c][scorer]["jaccard"] for c in shared]))
        theirs = float(np.mean([ref_cells[c][scorer]["jaccard"] for c in shared]))
        repro[scorer] = {"ours": ours, "banked": theirs, "abs_diff": abs(ours - theirs),
                         "n_cells": len(shared)}
        at.add("m8_reproduction_mean_jaccard", value=ours, model_id=WT.PG1_REPO,
               model_sha=base_sha, n=len(shared),
               control_id=MC.STRUCTURED if scorer != "predictor" else MC.TREATMENT,
               flags={"scorer": scorer, "banked_mean": theirs,
                      "abs_diff": repro[scorer]["abs_diff"], "tol": REPRO_TOL,
                      "reference": M8_REFERENCE})

    beats_ours = sum(1 for c in shared if per_cell[c]["beats_activation_baseline"])
    beats_ref = sum(1 for c in shared if ref_cells[c]["beats_activation_baseline"])

    # The named target (freeze A6): the ACTIVATION-MAGNITUDE mean, the incumbent that won.
    gates.require("g4_reproduces_m8_activation_baseline",
                  repro["activation_magnitude"]["abs_diff"] <= REPRO_TOL,
                  **repro["activation_magnitude"], tol=REPRO_TOL,
                  note=("the spec's 'Jaccard 0.416-class' target is the ACTIVATION-MAGNITUDE mean "
                        "(freeze A6), not the predictor's 0.3068"))
    gates.record("g4_reproduces_m8_predictor", repro["predictor"]["abs_diff"] <= REPRO_TOL,
                 **repro["predictor"], tol=REPRO_TOL)
    gates.record("g4_reproduces_m8_refuter_direction",
                 beats_ours <= len(shared) / 2 and beats_ref <= len(shared) / 2,
                 beats_ours=beats_ours, beats_banked=beats_ref, n_cells=len(shared),
                 note="M-8's refuter fired in the banked run; the reproduction must fire too")

    info = at.to_parquet()
    man.write()
    verdict = MP.check_verdict(ARM, "REPRODUCED"
                               if repro["activation_magnitude"]["abs_diff"] <= REPRO_TOL
                               else "NOT-REPRODUCED")
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"],
        "base_repo": WT.PG1_REPO, "base_sha": base_sha, "n_frozen_windows": a.n_seq,
        "types": types, "n_layers": len(layers), "k": a.k,
        "n_cells_banked": len(per_cell), "n_cells_compared": len(shared),
        "reproduction": repro, "beats_activation_ours": beats_ours,
        "beats_activation_banked": beats_ref,
        "vectors_banked": ["act_rms", "grad_mag", "grad_cons"],
        "atlas": info, "reference": M8_REFERENCE, "per_cell": per_cell})
    print("G4_OK " + json.dumps({"verdict": verdict, "n_cells": len(per_cell),
                                 "repro": {k: round(v["abs_diff"], 5) for k, v in repro.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
