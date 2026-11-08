"""M-D2 — the core-relative frame retry, with a dictionary trained for this model (queue D2).

W0-13 asked whether feature coordinates expressed in the core frame are more stable across
fine-tunes than the same features in raw coordinates or in twin frames. It could not be answered
because the only public dictionaries for this residual stream are Gemma Scope's, which are trained
on gemma-2 and are out of distribution on PG1's gemma-1 decoder. So this arm TRAINS one.

    THE DICTIONARY. A BatchTopK SAE on PG1's layer-12 residual, trained on the house blank-image
    forward over wikitext windows, using `diff_sae.train_diff_sae` — the same trainer, the same
    recipe and the same quality readout the W9 Diff-SAE work used, so "a dictionary for this model"
    is a known object rather than a new one.

    THE QUESTION, MADE ANSWERABLE. Features cannot be tracked across fine-tunes by retraining a
    dictionary per model — feature matching is its own unsolved problem and would dominate the
    answer. So the dictionary is FROZEN and the fine-tune enters through the activations:

        f      = SAE(h_base)                      the base model's features
        f_full = SAE(h_ft)                        the fine-tune's features
        f_U    = SAE(h_base + P_U (h_ft - h_base))   only the frame's component of the change

    and the statistic is share_U = drift(f_U) / drift(f_full): how much of the feature drift a
    frame's component of the activation change accounts for. A frame that carries all the drift
    scores 1; one that carries none scores 0.

    REFUTER (pre-logged): the core frame's share is inside the matched-random band.

WHY THIS IS THE HONEST VERSION OF W0-13's QUESTION. "More stable in the core frame" is only
meaningful against frames of the SAME dimension, so every frame here is 32-dimensional and the
battery includes three Haar draws, the consensus extremes, the singular extremes and the
complement. Raw coordinates are reported too, as the full-dimensional reference the shares are
normalised by.

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
from infra.pg import mercator_program as P

ARM = "M-D2"
LAYER = 12
FRAMES = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
          "haar32_s0", "haar32_s1", "haar32_s2")
# PG1 fine-tunes from the same family the census uses, small and diverse in task.
FINETUNES = ("google/paligemma-3b-ft-cococap-224", "google/paligemma-3b-ft-vqav2-224",
             "google/paligemma-3b-ft-textvqa-224", "google/paligemma-3b-ft-refcoco-seg-224")


def main():
    ap = argparse.ArgumentParser(description="M-D2 — core-relative frame retry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--n-windows", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--sae-steps", type=int, default=4000)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--n-eval-windows", type=int, default=64)
    ap.add_argument("--finetunes", default=",".join(FINETUNES))
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()
    # dictionary_learning/training.py imports wandb at module top even with use_wandb=False, and
    # the AMI does not carry it. Installed at second zero rather than after the harvest: the first
    # run died on the import having already built 33023 training and 8256 evaluation activations
    # (aws-pg gotcha 12; the same shim DARWIN's SAE arms use).
    from infra.pg.diff_harvest import ensure_wandb
    ensure_wandb()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fts = [f.strip() for f in a.finetunes.split(",") if f.strip()]
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w1") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_d2", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import gc
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    from infra.pg import diff_sae as DS
    from infra.aws_pg.jobs.invariant_core_func import (_blank_image, _decoder_layers,
                                                       _load_full_pg1, _locate_text_block,
                                                       _processor_text_ids)
    from infra.aws_pg.jobs.invariant_core import _delete_cache

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    tok = processor.tokenizer
    blank = _blank_image(128)
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    U = {b: torch.as_tensor(WT.load_bank(bank_dir, "q_proj", a.layer, b), dtype=torch.float32)
         for b in FRAMES}

    # ---- the fixed token windows, self-aligned through the verified forward -------------------
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    raw = "\n\n".join(t for t in ds["text"] if t.strip())
    ids_all = tok(raw, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n_win = min(a.n_windows + a.n_eval_windows, ids_all.shape[0] // a.seq_len)
    windows = []
    for w in range(n_win):
        text = tok.decode(ids_all[w * a.seq_len:(w + 1) * a.seq_len], skip_special_tokens=True)
        # Tokenise the string the PROCESSOR builds, newline included — see _processor_text_ids.
        # `tok(text)` is right only when the trailing newline does not merge into the final token,
        # and this job drew a window where it does (M-D51, the same defect as M-D45 in M-E2).
        win_ids = _processor_text_ids(tok, text)
        if win_ids.shape[0] < 16:
            continue
        inp = processor(text=[text], images=[blank], return_tensors="pt", padding=False)
        st = _locate_text_block(inp["input_ids"][0], win_ids)
        windows.append({"start": int(st), "n": int(win_ids.shape[0]),
                        "input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"],
                        "pixel_values": inp["pixel_values"].to(torch.bfloat16)})
    assert len(windows) > a.n_eval_windows + 16, f"only {len(windows)} usable windows"
    train_w = windows[: len(windows) - a.n_eval_windows]
    eval_w = windows[len(windows) - a.n_eval_windows:]
    print(f"[M-D2] {len(train_w)} SAE-training windows, {len(eval_w)} evaluation windows",
          flush=True)

    def harvest(model, wins):
        """Layer-`layer` residual over the located text block, stacked (N, d) on CPU fp32."""
        dec = _decoder_layers(model)
        buf = {}

        def hook(_m, args):
            buf["h"] = args[0].detach().to(torch.float32)

        hd = dec[a.layer].self_attn.q_proj.register_forward_pre_hook(hook)
        rows = []
        with torch.inference_mode():
            for win in wins:
                model(input_ids=win["input_ids"].to("cuda"),
                      attention_mask=win["attention_mask"].to("cuda"),
                      pixel_values=win["pixel_values"].to("cuda"))
                rows.append(buf["h"][0, win["start"]:win["start"] + win["n"], :].cpu())
        hd.remove()
        return torch.cat(rows, dim=0)

    base = _load_full_pg1(base_sha, "cuda").eval()
    acts_train = harvest(base, train_w)
    acts_eval = harvest(base, eval_w)
    print(f"[M-D2] harvested {tuple(acts_train.shape)} train / {tuple(acts_eval.shape)} eval",
          flush=True)

    ae, cfg = DS.train_diff_sae(acts_train, a.layer, a.seed, "cuda",
                                expansion=a.expansion, steps=a.sae_steps)
    q = DS.diff_sae_quality(ae, acts_eval, "cuda")
    at.add("dictionary_fvu", value=float(q["fvu"]), model_id=WT.PG1_REPO, model_sha=base_sha,
           flags={**cfg, **{k: v for k, v in q.items() if isinstance(v, (int, float))},
                  "layer": a.layer, "trainer": "diff_sae.train_diff_sae (BatchTopK)"})
    # A DICTIONARY THAT DOES NOT RECONSTRUCT IS NOT A FEATURE BASIS. Everything below reads
    # feature drift, which is meaningless if the features do not explain the activations.
    dict_ok = bool(q["fvu"] < 0.5)
    gates.record("d2_dictionary_reconstructs", dict_ok, fvu=float(q["fvu"]), threshold=0.5,
                 outcome=None if dict_ok else M.VOID_INSTRUMENT,
                 note=("FVU is the fraction of activation variance the dictionary fails to "
                       "explain; above 0.5 the features are not a description of this stream"))
    assert dict_ok, (
        f"the trained dictionary has FVU {q['fvu']:.4f} >= 0.5; feature drift measured through it "
        f"would be drift of something that is not describing the activations")
    print(f"[M-D2] dictionary FVU {q['fvu']:.4f}, dict_size {cfg['dict_size']}", flush=True)

    ae = ae.to("cuda").eval()

    @torch.no_grad()
    def encode(h):
        # `use_threshold=True` is the inference path BatchTopK's own quality gate uses; encoding
        # with the training-time top-k rule instead would make the feature vector depend on the
        # rest of the batch, which is not a property a per-token drift can have.
        return ae.encode(h.to("cuda"), use_threshold=True).to(torch.float32)

    f_base = encode(acts_eval)
    base_scale = float(f_base.abs().mean()) + 1e-12

    def drift(f_other):
        return float((f_other - f_base).abs().mean()) / base_scale

    del base
    gc.collect()
    torch.cuda.empty_cache()

    results = {}
    for ft in fts:
        ft_sha = api.model_info(ft).sha
        # Bounded, counted transport retry: this loop streams several multi-gigabyte checkpoints
        # and the first run died on a truncated body (2.74 GB of 4.95 GB) AFTER training the
        # dictionary to FVU 0.4372 — hours of work thrown away by one dropped connection.
        m, n_retry = M.retry_transport(
            lambda: PaliGemmaForConditionalGeneration.from_pretrained(
                ft, revision=ft_sha, torch_dtype=torch.bfloat16, device_map="cuda",
                attn_implementation="eager").eval(),
            what=ft)
        if n_retry:
            print(f"[M-D2] {ft} loaded after {n_retry} transport retries", flush=True)
        acts_ft = harvest(m, eval_w)
        del m
        gc.collect()
        torch.cuda.empty_cache()
        _delete_cache(ft)
        assert acts_ft.shape == acts_eval.shape, (
            f"{ft}: harvested {tuple(acts_ft.shape)} against base {tuple(acts_eval.shape)}; the "
            f"token blocks must line up for a per-token drift to mean anything")
        dh = acts_ft - acts_eval
        full = drift(encode(acts_ft))
        assert full > 0, f"{ft}: the fine-tune produced identical features; nothing to attribute"
        cell = {"finetune": ft, "sha": ft_sha, "drift_full": full,
                "activation_change_norm": float(dh.norm())}
        for b in FRAMES:
            Ub = U[b]
            proj = (dh @ Ub) @ Ub.T                       # P_U applied to the activation change
            share = drift(encode(acts_eval + proj)) / full
            energy = float((proj ** 2).sum() / (dh ** 2).sum())
            cell[b] = {"drift_share": share, "energy_share": energy,
                       "share_over_energy": share / energy if energy > 0 else float("nan")}
            at.add("frame_drift_share", value=share, bank_id=b, model_id=ft, model_sha=ft_sha,
                   flags={"finetune": ft, "layer": a.layer, "energy_share": energy,
                          "drift_full": full, "dict_size": cfg["dict_size"],
                          "definition": ("drift(SAE(h_base + P_U dh)) / drift(SAE(h_ft)); how "
                                         "much of the feature drift this frame's component of "
                                         "the activation change accounts for")})
            at.add("frame_energy_share", value=energy, bank_id=b, model_id=ft, model_sha=ft_sha,
                   flags={"finetune": ft, "layer": a.layer,
                          "definition": "||P_U dh||^2 / ||dh||^2, the activation-side reference"})
        results[ft] = cell
        print(f"[M-D2] {ft.split('/')[-1]:32s} full_drift={full:.4f}  " +
              "  ".join(f"{b.replace('32','')}={cell[b]['drift_share']:.3f}"
                        for b in ("core32", "anti32", "haar32_s0")), flush=True)
        M.s3_sync_incremental(out)

    at.check_controls_per_statistic(["frame_drift_share", "frame_energy_share"])
    core = float(np.mean([results[f]["core32"]["drift_share"] for f in fts]))
    twins = [float(np.mean([results[f][b]["drift_share"] for f in fts]))
             for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
    band = [min(twins), max(twins)]
    outside = bool(core < band[0] or core > band[1])
    # A TWO-SIDED GATE MUST SAY WHICH SIDE IT LANDED ON. "Outside the band" is satisfied by the
    # core carrying MORE drift than a random frame and by its carrying LESS, and those are opposite
    # findings — a bare PASS reads as support for whichever one the reader expected. The direction
    # is the whole content of the result, so it is a field and not only a note (M-D69, the same
    # shape as M-D34's magnitude-only correlation gate).
    gates.record("d2_core_frame_outside_matched_random_band", outside,
                 core=core, twin_band=band,
                 direction=("above" if core > max(band) else
                            "below" if core < min(band) else "inside"),
                 by_frame={b: float(np.mean([results[f][b]["drift_share"] for f in fts]))
                           for b in FRAMES},
                 note=("the refuter fires when this is False. Below the band means the core "
                       "carries LESS of the feature drift than a random frame of the same "
                       "dimension; above means it carries more"))
    # The energy-normalised reading, because a frame that holds more of the activation change will
    # account for more of the drift for that reason alone.
    core_ratio = float(np.mean([results[f]["core32"]["share_over_energy"] for f in fts]))
    twin_ratios = [float(np.mean([results[f][b]["share_over_energy"] for f in fts]))
                   for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
    gates.record("d2_core_frame_outside_band_per_unit_energy",
                 bool(core_ratio < min(twin_ratios) or core_ratio > max(twin_ratios)),
                 core=core_ratio, twin_band=[min(twin_ratios), max(twin_ratios)],
                 direction=("above" if core_ratio > max(twin_ratios) else
                            "below" if core_ratio < min(twin_ratios) else "inside"),
                 note=("drift share divided by energy share: a frame that simply holds more of "
                       "the change accounts for more drift for that reason alone"))
    sur.add(arm=ARM, expected_ref=expected_ref, cell="core32",
            statistic="frame_drift_share/core32",
            s=M.surprise(core, float(np.mean(twins)), twins))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "layer": a.layer, "dictionary": cfg, "dictionary_quality": q,
                  "n_train_tokens": int(acts_train.shape[0]),
                  "n_eval_tokens": int(acts_eval.shape[0]), "finetunes": fts,
                  "cells": results, "core_drift_share": core, "twin_band": band,
                  "core_outside_band": outside,
                  "core_share_over_energy": core_ratio, "twin_share_over_energy": twin_ratios})
    print("D2_OK " + json.dumps({"rows": info["n_rows"], "fvu": float(q["fvu"]),
                                 "core": core, "twin_band": band, "outside": outside}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
