"""M-B1 — the OUTPUT-SIDE census (queue item B1). CPU.

Every measurement HUMBOLDT and DARWIN made is input-space: the consensus Gram is dW^T dW, an n x n
object on the coordinates a matrix READS. The row-space consensus dW dW^T — the m x m object on the
coordinates a matrix WRITES — has never been computed. This job is that half.

The two objects share a spectrum up to |m-n| zeros, so "the spectra agree" is a tautology and is
not a finding. What the row side alone determines is WHERE the suppressed directions sit in output
coordinates, how they spread leverage over rows, and whether they line up across layers. Those are
what this job records, each against the full twin battery in the row space.

REFUTER (spec §4): of "second sink" — bottom row-space at matched-rotation-null levels everywhere.

PARTITIONING. The Grams are m x m and m is the OUTPUT width, which for gate_proj/up_proj is 16384:
2.1 GB per cell, 38.7 GB over 18 layers. So the type list is partitioned across runs and each run
asserts its own memory budget BEFORE streaming 24 checkpoints:

  --types q_proj,k_proj,v_proj,o_proj,down_proj   ~1.8 GB   r7i.2xlarge
  --types gate_proj                               38.7 GB   r7i.4xlarge
  --types up_proj                                 38.7 GB   r7i.4xlarge

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as W
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P
from infra.pg import mercator_rowgram as RG

ARM = "M-B1"
FULL_BATTERY = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2")
# Statistics whose claim is about COORDINATES and therefore carry both bases (§3).
COORD_CLAIMS = ("row_leverage_max", "row_leverage_participation_ratio", "row_spike_count")


def _spike_count(lev, mult: float = 5.0) -> float:
    """Coordinates whose leverage exceeds `mult` x the flat expectation k/n. Same rule DARWIN's
    census used, reused so the row-side count is comparable to the input-side one."""
    n = lev.size
    k = float(lev.sum())
    return float((lev > mult * (k / n)).sum())


def main():
    ap = argparse.ArgumentParser(description="M-B1 output-side (row-space) consensus census")
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="q_proj,k_proj,v_proj,o_proj,down_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--budget-gb", type=float, default=45.0,
                    help="memory ceiling for the resident Grams; asserted BEFORE streaming")
    ap.add_argument("--max-fts", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    H.hf_setup()          # FIRST: huggingface_hub freezes HF_HUB_OFFLINE at import time

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    types = [t.strip() for t in a.types.split(",") if t.strip()]
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    for t in types:
        assert t in H.TYPES, f"unknown type {t}"
    if a.smoke:
        types, layers = types[:1], layers[:2]

    from huggingface_hub import HfApi
    api = HfApi()
    base_sha = api.model_info(W.PG1_REPO).sha
    repos_full = W.ft_cloud(api)
    repos = repos_full if a.max_fts <= 0 else repos_full[:a.max_fts]
    shas = {r: api.model_info(r).sha for r in repos}
    cloud_complete = len(repos) == len(repos_full)

    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w0") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id=f"m_b1_{'_'.join(types)}", register=meta["register"],
                 wave=meta["wave"], expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    # The smoke that proves the estimator can see a planted row-space block AND stays at chance on
    # noise. Run BEFORE the 24-checkpoint stream, so an instrument failure costs seconds.
    smoke = RG.planted_rowspace_null()
    gates.require("b1_planted_rowspace_smoke",
                  smoke["planted_capture"] > 0.9
                  and abs(smoke["null_capture"] - smoke["chance_capture"]) < 0.05, **smoke)

    # Memory budget, asserted before any download.
    out_dims = {"q_proj": 2048, "k_proj": 256, "v_proj": 256, "o_proj": 2048,
                "gate_proj": 16384, "up_proj": 16384, "down_proj": 2048}
    budget = RG.assert_fits(types, layers, out_dims, budget_gb=a.budget_gb)
    gates.record("b1_memory_budget", True, **budget)

    manifest = {"program": M.PROGRAM, "arm": ARM, "register": meta["register"], "wave": "W0",
                "queue_item": meta["queue_item"], "provenance": M.provenance(),
                "base_repo": W.PG1_REPO, "base_sha": base_sha, "n_fine_tunes": len(repos),
                "n_fine_tunes_full": len(repos_full), "cloud_complete": cloud_complete,
                "fine_tunes": repos, "fine_tune_shas": shas, "types": types, "layers": layers,
                "expected_ref": expected_ref, "seed": a.seed, "budget": budget,
                "side": RG.SIDE, "started_utc": M.utc_now(),
                "input_side_note": ("this arm measures dW dW^T. The eigenvalue SPECTRUM matches "
                                    "the input-side Gram's up to |m-n| zeros by construction, so "
                                    "spectral agreement is not evidence; the coordinate-side "
                                    "statistics are what the row side alone determines")}
    M.write_json(out, "manifest.json", manifest)
    assert cloud_complete or a.smoke, (
        "the consensus object is DEFINED by the cloud it sums over; a truncated cloud is a "
        "different matrix and its banks are not comparable to the input-side ones")

    import torch
    G, W0, fro2_by, n_retries = RG.stream_row_grams(repos, shas, base_sha, types,
                                                    layers)
    print(f"[M-B1] streamed {len(repos)} fine-tunes over {len(G)} cells", flush=True)

    n_banked, results = 0, {}
    for (t, L), g in sorted(G.items()):
        m = g.shape[0]
        seed = H.cell_seed(a.seed, ARM, t, L)
        # ONLY THE EXTREMES. The banks use 2k of the m eigenvectors and nothing in between, and a
        # full eigendecomposition of a 16384-wide row Gram costs minutes per cell — measured
        # overrunning the budget on the first attempt at this arm. The subset driver returns the
        # same vectors; `full_lam` is None for the wide types and that absence is recorded rather
        # than filled with a truncated spectrum.
        lb, Vb, lt, Vt, trace, full_lam = RG.eig_extremes(g, H.BANK_K)
        # W0 holds fp32 torch VIEWS into the base state dict; the row-side sigma twins are the
        # left singular vectors, computed in fp64 like every other bank in the program.
        banks = RG.build_row_banks_from_extremes(Vb, Vt, W0[(t, L)].to(torch.float64).numpy(),
                                                 k=H.BANK_K, seed_base=seed)
        RG.save_row_banks(out, t, L, banks)
        n_banked += 1

        av = RG.row_avoidance_from_extremes(lb, trace, m, H.BANK_K)
        at.add("row_bottom_k_energy_fraction", value=av["bottom_k_energy_fraction"], layer=L,
               type=t, bank_id="core32", model_id=W.PG1_REPO, model_sha=base_sha,
               flags={**av, "side": RG.SIDE, "n_fine_tunes": len(repos)})
        at.add("row_avoidance_ratio", value=av["ratio"], layer=L, type=t, bank_id="core32",
               model_id=W.PG1_REPO, model_sha=base_sha,
               flags={"flat_expectation": av["flat_expectation"], "side": RG.SIDE})

        # The full twin battery, per cell, on the coordinate claims — checked PER STATISTIC.
        cell = {}
        for bid, U in banks.items():
            lev = H.leverage(U)
            for stat, val in (("row_leverage_max", float(lev.max())),
                              ("row_leverage_participation_ratio", H.participation_ratio(lev)),
                              ("row_spike_count", _spike_count(lev))):
                at.add(stat, value=val, layer=L, type=t, bank_id=bid, model_id=W.PG1_REPO,
                       model_sha=base_sha, basis=M.BASIS_COORD,
                       flags={"side": RG.SIDE, "m": int(m), "k": int(H.BANK_K)})
            # §3 dual basis: the SAME bank conjugated by one random reference basis. Leverage is a
            # coordinate statistic, so it MUST move — a random reference basis destroys coordinate
            # structure — and the reference row is what shows the coordinate row was not an
            # artifact of the estimator.
            Q = M.random_reference_basis(m, seed)
            Ur = Q @ U
            levr = H.leverage(Ur)
            for stat, val in (("row_leverage_max", float(levr.max())),
                              ("row_leverage_participation_ratio", H.participation_ratio(levr)),
                              ("row_spike_count", _spike_count(levr))):
                at.add(stat, value=val, layer=L, type=t, bank_id=bid, model_id=W.PG1_REPO,
                       model_sha=base_sha, basis=M.BASIS_RANDOM_REF,
                       flags={"side": RG.SIDE, "m": int(m), "reference_basis_seed": seed})
            cell[bid] = {"leverage_max": float(lev.max()),
                         "participation_ratio": H.participation_ratio(lev),
                         "spike_count": _spike_count(lev),
                         "ref_leverage_max": float(levr.max()),
                         "ref_spike_count": _spike_count(levr)}
        # The REFUTER's own comparison, per cell: core32 against its haar band.
        haar = [cell[b]["leverage_max"] for b in ("haar32_s0", "haar32_s1", "haar32_s2")]
        s = M.surprise(cell["core32"]["leverage_max"], float(np.mean(haar)), haar)
        sur.add(arm=ARM, expected_ref=expected_ref, cell=f"{t}/L{L}",
                statistic="row_leverage_max/core32", s=s)
        results[f"{t}/L{L}"] = {"avoidance": av, "banks": cell,
                                "core_over_haar_leverage_max":
                                    cell["core32"]["leverage_max"] / float(np.mean(haar))}
        # The full spectrum is banked where it was affordable; where it was not, the extremes and
        # the exact trace are banked instead and the row says which it is.
        spec_name = f"{ARM}__lam__{t}__L{L}"
        if full_lam is not None:
            at.add("row_gram_spectrum", vector_ref=at.vector(spec_name, full_lam),
                   layer=L, type=t, model_id=W.PG1_REPO, model_sha=base_sha,
                   flags={"side": RG.SIDE, "m": int(m), "spectrum": "full"})
        else:
            at.add("row_gram_spectrum_extremes",
                   vector_ref=at.vector(spec_name, np.concatenate([lb, lt])),
                   layer=L, type=t, model_id=W.PG1_REPO, model_sha=base_sha,
                   flags={"side": RG.SIDE, "m": int(m), "spectrum": "bottom_k_then_top_k",
                          "k": int(H.BANK_K), "trace": float(trace),
                          "why": ("a full eigendecomposition of an m=%d row Gram is not "
                                  "affordable per cell; the banks use only the extremes" % m)})
        print(f"[M-B1] {t} L{L}: m={m} avoidance_ratio={av['ratio']:.4f} "
              f"core/haar leverage_max={results[f'{t}/L{L}']['core_over_haar_leverage_max']:.3f}",
              flush=True)
        # CRASH-SAFETY, per cell. A gate_proj/up_proj cell is an m=16384 eigendecomposition and
        # costs minutes; the reaper TERMINATES the instance at the deadline rather than signalling
        # it, so without this the EXIT trap is the only upload and an overrun loses every cell the
        # run had already finished. One sync of a nearly-unchanged directory is the price.
        M.s3_sync_incremental(out)

    # The battery check runs on the banks ACTUALLY written, per statistic — passing the literal
    # FULL_BATTERY set to check_controls would be a check that cannot fail.
    at.check_controls_per_statistic(list(COORD_CLAIMS))
    at.check_dual_basis(list(COORD_CLAIMS))
    gates.require("b1_all_cells_banked", n_banked == len(types) * len(layers),
                  n_banked=n_banked, expected=len(types) * len(layers))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, "M-B1_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "manifest": manifest, "smoke": smoke, "cells": results,
                  "download_retries": n_retries,
                  "n_download_retries": int(sum(n_retries.values()))})
    M.publish_banks(out)
    print("M-B1_OK " + json.dumps({"rows": info["n_rows"], "cells": n_banked}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
