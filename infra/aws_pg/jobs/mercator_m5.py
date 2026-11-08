"""M-5 — sink-aware gradient compression (queue item 5). Wave P, Register B.

    THE ENGINEERING CLAIM THIS TESTS. M-Q stage 2 and M-A3 stage 2 both found the same thing from
    different directions: the core absorbs a large share of the gradient (~5x isotropic) and Adam
    delivers very little motion there (delivery ratio ~0.2 against ~7 for a random subspace). If
    that is right, gradient bytes spent on sink coordinates are bytes spent on motion that will not
    happen — so a top-k compressor that EXCLUDES the sink before choosing its k entries should,
    at the same bandwidth, spend those slots on coordinates that do move.

    REFUTER (pre-logged): sink-excluded top-k is no better than naive top-k at matched bandwidth,
    or no better than the twin-excluded control.

THE COMPRESSOR IS `mercator_compress.compress`, IMPORTED UNCHANGED — three arms at matched
bandwidth, plus a dense reference:

    dense             no compression — the ceiling every compressor is measured against
    naive             top-k by |g|, the incumbent
    sink_excluded     top-k by |g| after removing the sink COLUMNS
    random_excluded   the same, on an equal number of random columns drawn disjoint from the sink
                      — THE control, because "excluding some columns helps" is a different claim
                      from "excluding the SINK helps"

The exclusion is per COLUMN because the sink is a set of residual-stream coordinates and a weight
matrix's columns are exactly its residual inputs. ERROR FEEDBACK is part of the instrument:
untransmitted gradient accumulates rather than being discarded, without which the naive arm would
be artificially bad and the comparison flattering.

BANDWIDTH IS COUNTED, NOT ASSUMED — the instrument asserts the arms transmit identical entry
counts and refuses a k that exceeds what is selectable after excluding. The excluding arms carry a
ONE-TIME index payload (which columns were removed, ints not floats), reported as a number of
steps'-worth of payload rather than waved at: a compressor whose fixed cost exceeds the run's
savings is not a compressor, and the row says which regime it is in.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import darwin_train as T
from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ARM = "M-5"
PPL_CORPUS = "wikitext2:window512"
TASK_CORPUS = "codeparrot-clean-valid:completion200"
# The bandwidth ladder, as a fraction of each matrix's entries kept per step.
# BUILT BY geomspace, NOT WRITTEN OUT BY HAND. The hand-picked (0.001, 0.01, 0.05, 0.2) reads as
# log-spaced and is not — its ratios are 10, 5, 4 — so `check_dose_grid` refused it at second zero
# and took the whole four-step pipeline down with it. Constructing the grid makes the property true
# by definition instead of by eye.
KEEP_FRACTIONS = tuple(round(float(x), 6) for x in np.geomspace(0.001, 0.2, 4))
# The sink is a set of residual-stream COORDINATES, and a weight matrix's columns are exactly its
# residual inputs — so the exclusion is per column. The coordinates are the spike set of the core
# bank at the frozen multiple the re-census uses, recomputed here from the bank rather than read
# from a list, so the arm and the census cannot disagree about what the sink is.
SPIKE_MULT = 5.0


def spike_coords(U, mult: float = SPIKE_MULT):
    """Input coordinates whose leverage in the bank exceeds `mult` times the flat expectation."""
    lev = H.leverage(U)
    return np.flatnonzero(lev > mult * (U.shape[1] / U.shape[0]))


def main():
    ap = argparse.ArgumentParser(description="M-5 — sink-aware gradient compression")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-fractions", default=",".join(str(f) for f in KEEP_FRACTIONS))
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="12,13,14,15,16,17")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fracs = [float(x) for x in a.keep_fractions.split(",") if x.strip()]
    M.check_dose_grid(fracs, arm=ARM)
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()]
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_wp") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_5", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)
    optinfo = M.check_optimizer(ARM, "darwin_train.ExactAdam", offload_exact_fp32=False)

    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.aws_pg.jobs.invariant_core_func import (_decoder_layers, _load_full_pg1,
                                                       _ppl_blankimg)
    from infra.aws_pg.jobs.darwin_w1_u import build_batches, eval_task_loss, text_task_rows
    from infra.pg import mercator_compress as CP

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    for p in model.parameters():
        p.requires_grad_(False)
    mods = T.trainable_modules(_decoder_layers(model), types=types, layers=layers)
    for mod in mods.values():
        mod.weight.requires_grad_(True)
    pristine = {k: m.weight.detach().to(torch.float32).clone() for k, m in mods.items()}
    train_rows, eval_rows = text_task_rows(a.n_train, a.n_eval, a.seed)
    train_b = build_batches(processor, train_rows, "cuda")
    eval_b = build_batches(processor, eval_rows, "cuda")
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    sink = {k: spike_coords(WT.load_bank(bank_dir, k[0], k[1], "core32")) for k in mods}
    n_sink = {k: int(len(v)) for k, v in sink.items()}
    assert all(v > 0 for v in n_sink.values()), (
        f"some cell has no spike coordinates at mult {SPIKE_MULT}: "
        f"{[k for k, v in n_sink.items() if v == 0]}; there is nothing for the arm to exclude")
    base_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
    n_entries = int(sum(m.weight.numel() for m in mods.values()))
    # THE ONE-TIME PAYLOAD IS A COORDINATE LIST, not a basis: the receiver needs to know WHICH
    # columns were excluded, which is int32 per coordinate, not a 32 x n float block.
    index_ints = int(sum(n_sink.values()))
    print(f"[M-5] base ppl={base_ppl:.5f}; {n_entries} trainable entries across {len(mods)} cells; "
          f"sink columns per cell {min(n_sink.values())}-{max(n_sink.values())}; "
          f"one-time index payload {index_ints} ints", flush=True)

    def run(arm, frac):
        """One fine-tune under one compression arm.

        `mercator_compress.compress` is the estimator, imported unchanged: it carries the error
        feedback (untransmitted gradient accumulates rather than being discarded — leaving that out
        would make the naive arm artificially bad), the column exclusion, and the assertion that k
        does not exceed what is selectable after excluding.
        """
        for k, m in mods.items():
            m.weight.data.copy_(pristine[k].to(m.weight.dtype))
        opt = T.ExactAdam([m.weight for m in mods.values()], lr=a.lr)
        rng = np.random.default_rng(H.cell_seed(a.seed, ARM, arm, f"{frac:g}"))
        resid = {k: None for k in mods}
        sent, share_rows, k_used = 0, [], 0
        model.train()
        for step in range(a.steps):
            b = train_b[int(rng.integers(0, len(train_b)))]
            opt.zero_grad()
            model(input_ids=b["input_ids"], labels=b["labels"]).loss.backward()
            if arm != "dense":
                with torch.no_grad():
                    for key, m in mods.items():
                        g = m.weight.grad.to(torch.float32).cpu().numpy().astype(np.float64)
                        if step == 0:
                            share_rows.append(CP.sink_energy_share(g, sink[key]))
                        keep = max(1, int(round(frac * g.size)))
                        r = CP.compress(g, keep, arm=arm, sink_coords=sink[key],
                                        seed=H.cell_seed(a.seed, ARM, str(key), f"{frac:g}"),
                                        residual=resid[key])
                        resid[key] = r["residual"]
                        k_used = r["k"]
                        m.weight.grad = torch.as_tensor(
                            r["sent"], device="cuda", dtype=m.weight.grad.dtype)
                        sent += r["k"]
            opt.step()
        model.eval()
        ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
        task = eval_task_loss(model, eval_b)
        motion = float(np.sqrt(sum(
            float(((m.weight.detach().to(torch.float32) - pristine[k]) ** 2).sum())
            for k, m in mods.items())))
        per_step = sent / a.steps if arm != "dense" else n_entries
        excl = 0 if arm in ("dense", "naive") else index_ints
        return {"arm": arm, "k_per_cell": int(k_used), "keep_fraction":
                frac if arm != "dense" else 1.0, "ppl": ppl,
                "task_loss": task, "total_motion": motion,
                "values_sent_per_step": per_step,
                "one_time_index_ints": excl,
                "index_cost_in_steps": (0.0 if excl == 0 else excl / max(per_step, 1)),
                "sink_energy_share": (float(np.mean([r["ratio"] for r in share_rows]))
                                      if share_rows else None),
                "arm_kind": arm}

    cells = {}
    dense = run("dense", 1.0)
    cells["dense"] = dense
    at.add("compression_task_loss", value=dense["task_loss"], corpus_id=TASK_CORPUS,
           model_id=WT.PG1_REPO, model_sha=base_sha, incumbent_id="dense", flags=dense)
    print(f"[M-5] dense: task={dense['task_loss']:.4f} ppl={dense['ppl']:.4f}", flush=True)
    for frac in fracs:
        for arm in CP.ARMS:
            c = run(arm, frac)
            cells[f"{arm}@{frac:g}"] = c
            bid = "core32" if arm == "sink_excluded" else None
            at.add("compression_task_loss", value=c["task_loss"], dose=frac, bank_id=bid,
                   corpus_id=TASK_CORPUS, model_id=WT.PG1_REPO, model_sha=base_sha,
                   incumbent_id="naive", flags={**c, "dense_task_loss": dense["task_loss"],
                                               "steps": a.steps, "lr": a.lr,
                                               "optimizer": optinfo})
            at.add("compression_ppl", value=c["ppl"], dose=frac, bank_id=bid,
                   corpus_id=PPL_CORPUS, model_id=WT.PG1_REPO, model_sha=base_sha,
                   flags={**c, "base_ppl": base_ppl, "dense_ppl": dense["ppl"]})
            print(f"[M-5] {arm:16s} keep={frac:<6g} task={c['task_loss']:.4f} "
                  f"ppl={c['ppl']:.4f} sent/step={c['values_sent_per_step']:.0f} "
                  f"index_cost={c['index_cost_in_steps']:.2f} steps "
                  f"sink_energy_ratio={c['sink_energy_share']}", flush=True)
        # The instrument's own structural control, on the k values the arms ACTUALLY used.
        CP.matched_bandwidth_check([{"arm": arm, "k": cells[f"{arm}@{frac:g}"]["k_per_cell"]}
                                    for arm in CP.ARMS])
        M.s3_sync_incremental(out)

    at.check_readout_corpus(["compression_task_loss", "compression_ppl"])
    # ---- the refuter, at every bandwidth ------------------------------------------------------
    by_frac = {}
    for frac in fracs:
        naive = cells[f"naive@{frac:g}"]["task_loss"]
        sk = cells[f"sink_excluded@{frac:g}"]["task_loss"]
        twin = cells[f"random_excluded@{frac:g}"]["task_loss"]
        by_frac[frac] = {"naive": naive, "sink_excluded": sk, "random_excluded": twin,
                         "beats_naive": bool(sk < naive), "beats_random": bool(sk < twin),
                         "recovered_vs_dense": ((naive - sk) / (naive - dense["task_loss"])
                                                if naive != dense["task_loss"] else float("nan"))}
    wins = [f for f, v in by_frac.items() if v["beats_naive"] and v["beats_random"]]
    gates.record("m5_sink_exclusion_beats_naive_and_random_at_some_bandwidth", bool(wins),
                 winning_fractions=wins, by_bandwidth=by_frac,
                 note=("both comparisons are required: beating naive top-k while not beating the "
                       "random-excluded control means 'excluding some columns helps', which is a "
                       "different claim"))
    # THE ARM'S OWN PREMISE, MEASURED rather than cited: if the sink's gradient-energy share is
    # not elevated on this model and this data, the motivation does not hold here.
    ratios = [cells[f"naive@{f:g}"]["sink_energy_share"] for f in fracs
              if cells[f"naive@{f:g}"]["sink_energy_share"] is not None]
    gates.record("m5_sink_energy_share_is_elevated",
                 bool(ratios and min(ratios) > 1.0), sink_energy_ratio=ratios,
                 note=("share of gradient energy in the sink columns over the flat expectation; "
                       "at or below 1.0 there is no concentration to exploit"))
    gates.record("m5_index_cost_is_amortised",
                 bool(all(cells[f"sink_excluded@{f:g}"]["index_cost_in_steps"] < a.steps
                          for f in fracs)),
                 index_cost_in_steps={f"{f:g}":
                                      cells[f"sink_excluded@{f:g}"]["index_cost_in_steps"]
                                      for f in fracs}, run_steps=a.steps,
                 note=("the excluding arms send the column index list once; if that costs more "
                       "than the run sends in total, the saving is overhead"))
    best = min(fracs, key=lambda f: by_frac[f]["sink_excluded"])
    sur.add(arm=ARM, expected_ref=expected_ref, cell=f"keep={best:g}",
            statistic="compression_task_loss/core32",
            s=M.surprise(by_frac[best]["sink_excluded"], by_frac[best]["random_excluded"],
                         [by_frac[f]["random_excluded"] for f in fracs]))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "types": list(types), "layers": layers, "steps": a.steps, "lr": a.lr,
                  "keep_fractions": fracs, "optimizer": optinfo, "base_ppl": base_ppl,
                  "n_trainable_entries": n_entries, "one_time_index_ints": index_ints,
                  "sink_columns_per_cell": {f"{k[0]}.{k[1]}": v for k, v in n_sink.items()},
                  "compressor": ("mercator_compress.compress, imported unchanged: error feedback, "
                                 "column exclusion, matched-bandwidth assertion"),
                  "dense": dense, "cells": cells, "by_bandwidth": by_frac,
                  "winning_fractions": wins})
    print("M5_OK " + json.dumps({"rows": info["n_rows"], "winning_fractions": wins,
                                 "by_bandwidth": {str(k): v["beats_naive"]
                                                  for k, v in by_frac.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
