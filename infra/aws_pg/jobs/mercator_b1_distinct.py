"""M-B1, second half — is the row-side suppressed subspace DISTINCT from the input-side one?

The first half measured the row side thoroughly and never compared it to the input side, which is
what M-B1's pre-logged direction is actually about: "a second, DISTINCT suppressed subspace — not
merely the input-side spectrum re-read".

    WHY THE OBVIOUS COMPARISON IS NOT THE ONE. Comparing the two banks by principal angles is a
    category error: for a reader like q_proj the INPUT space is the residual stream and the OUTPUT
    space is the head-structured query space. They happen to share a dimension for square types and
    are not the same space, so an overlap between them measures nothing.

    THE COMPARISON THAT IS DEFINED. Both sides produce ONE NUMBER per cell — the avoidance ratio,
    bottom-k energy fraction over the flat expectation, computed identically on the input Gram and
    on the row Gram of the SAME 24-fine-tune cloud. If the row side were the input spectrum re-read,
    those two numbers would move together across the cells. So the statistic is their rank
    correlation over all 90 cells, and its per-type breakdown.

    REFUTER (this half's share of M-B1's): row-side avoidance tracks input-side avoidance across
    cells, i.e. the row side carries no information the input side does not.

BOTH GRAMS COME FROM ONE STREAM. The fine-tunes are streamed once and both the input Gram
(dW^T dW) and the row Gram (dW dW^T) are accumulated in the same pass, so the two sides are the
same cloud and any difference between them is a difference of side.

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


def main():
    ap = argparse.ArgumentParser(description="M-B1 second half — input vs row distinctness")
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="q_proj,k_proj,v_proj,o_proj,down_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--n-fts", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_w0") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_b1_distinct", register=meta["register"],
                 wave=meta["wave"], expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import gc
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoConfig
    from infra.pg import darwin_census as DC
    from infra.pg.humboldt_weights import matrix_key
    from infra.aws_pg.jobs.invariant_core import _delete_cache

    api = HfApi()
    base_sha = api.model_info(W.PG1_REPO).sha
    geom = DC.base_geometry(AutoConfig.from_pretrained(W.PG1_REPO, revision=base_sha))
    accepted, log = DC.select_derivatives(api, W.PG1_REPO, geom, n_accept=a.n_fts, out=out)
    shas = {r["id"]: api.model_info(r["id"]).sha for r in accepted}
    # HEADER PREFLIGHT, the same one the census runs (M-D54). `select_derivatives` accepts on the
    # roster's own criteria and this job then LOADS each repo, so a checkpoint that cannot be
    # instantiated into the base architecture — a 768-wide vision tower against PG1's 1152 — takes
    # the job down mid-stream, after the earlier derivatives have already been downloaded and
    # accumulated. Validating from the shard headers costs kilobytes and turns that into a
    # recorded rejection.
    accepted, prevalid_log = DC.preflight_roster(W.PG1_REPO, base_sha, accepted, shas,
                                                 types, layers, tag="M-B1d")
    M.write_json(out, "preflight.json", prevalid_log)
    assert len(accepted) >= DC.MIN_FTS_FOR_CENSUS, (
        f"only {len(accepted)} of {len(prevalid_log)} derivatives survived the header preflight, "
        f"below the census threshold {DC.MIN_FTS_FOR_CENSUS}")
    keys = [(t, L) for t in types for L in layers]
    # M-B1's OWN LOADER, not darwin_census.load_decoder_matrices. The latter resolves a key by
    # suffix and PaliGemma's SigLIP tower carries the same `layers.N.self_attn.q_proj.weight` tail
    # as the LM decoder, so it fails loud on two hits — defect M-D6, met again here (M-D37). The
    # rowgram loader is already scoped to the language model and is what banked the 90 cells this
    # job is comparing against, so using it also guarantees the two sides read the same tensors.
    base_full, _r = RG._load_decoder_retrying(W.PG1_REPO, base_sha)
    base = {k: base_full[matrix_key(k[0], k[1])] for k in keys}
    print(f"[M-B1d] {len(accepted)} fine-tunes, {len(keys)} cells", flush=True)

    # BOTH GRAMS FOR EVERY CELL ARE LIVE AT ONCE, so the footprint is decided by the type list
    # before a single derivative is streamed — and for the MLP types it is enormous: down_proj's
    # input Gram is 16384^2 float64 = 2.1 GB PER LAYER, 39 GB across 18. Asking for all seven
    # types needs 118 GB. Without this check that arrives as a bare SIGKILL five minutes in, with
    # a STATUS of 137 and nothing in the log saying why (defect M-D41), so the arithmetic is done
    # here where it can name the type that does not fit.
    import os
    by_type = {}
    for k in keys:
        m_, n_ = int(base[k].shape[0]), int(base[k].shape[1])
        by_type[k[0]] = by_type.get(k[0], 0.0) + (n_ * n_ + m_ * m_) * 8
    need = sum(by_type.values())
    have = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    print("[M-B1d] Gram footprint: " + " ".join(f"{t}={v/1e9:.1f}GB"
                                                for t, v in sorted(by_type.items())) +
          f" | total {need/1e9:.1f}GB of {have/1e9:.1f}GB RAM", flush=True)
    assert need < 0.6 * have, (
        f"the two Grams for {sorted(by_type)} need {need/1e9:.1f} GB and this box has "
        f"{have/1e9:.1f} GB. Split the run by type — the 16384-wide MLP types cost "
        f"{2.1 * len(layers):.0f} GB each on their wide side and the 2048-square attention types "
        f"cost about 1 GB each, so they do not belong in one job")

    # ONE STREAM, BOTH GRAMS. Accumulating the input and row Grams in the same pass is what makes
    # the two sides the same cloud; two passes would let a dropped derivative differ between them.
    Gin = {k: np.zeros((base[k].shape[1],) * 2, dtype=np.float64) for k in keys}
    Grow = {k: np.zeros((base[k].shape[0],) * 2, dtype=np.float64) for k in keys}
    n_terms = 0
    for i, rec in enumerate(accepted):
        repo = rec["id"]
        ft_full, _n = RG._load_decoder_retrying(repo, shas[repo])
        ft = {k: ft_full[matrix_key(k[0], k[1])] for k in keys}
        for k in keys:
            assert ft[k].shape == base[k].shape, (
                f"{repo} {k}: {tuple(ft[k].shape)} vs base {tuple(base[k].shape)}")
            A = (ft[k].to(torch.float64) - base[k].to(torch.float64)).numpy()
            f2 = float((A ** 2).sum())
            if f2 == 0:
                continue
            Gin[k] += (A.T @ A) / f2
            Grow[k] += (A @ A.T) / f2
        n_terms += 1
        del ft, ft_full
        gc.collect()
        _delete_cache(repo)
        print(f"[M-B1d] {repo} ({i + 1}/{len(accepted)})", flush=True)
        M.s3_sync_incremental(out)
    assert n_terms >= DC.MIN_FTS_FOR_CENSUS, f"only {n_terms} usable derivatives"

    rows = []
    for k in keys:
        ain = RG.row_avoidance(Gin[k] / n_terms, k=H.BANK_K)
        arow = RG.row_avoidance(Grow[k] / n_terms, k=H.BANK_K)
        rows.append({"cell": f"{k[0]}/L{k[1]}", "type": k[0], "layer": k[1],
                     "input_avoidance": ain["ratio"], "row_avoidance": arow["ratio"],
                     "input_m": ain["m"], "row_m": arow["m"]})
        at.add("input_avoidance_ratio", value=ain["ratio"], layer=k[1], type=k[0],
               bank_id="core32", model_id=W.PG1_REPO, model_sha=base_sha,
               flags={**ain, "side": "input", "n_terms": n_terms})
        at.add("row_avoidance_ratio", value=arow["ratio"], layer=k[1], type=k[0],
               bank_id="core32", model_id=W.PG1_REPO, model_sha=base_sha,
               flags={**arow, "side": "row", "n_terms": n_terms})
        print(f"[M-B1d] {k[0]:10s} L{k[1]:<2d} input={ain['ratio']:.4f} "
              f"row={arow['ratio']:.4f}", flush=True)

    x = np.array([r["input_avoidance"] for r in rows])
    y = np.array([r["row_avoidance"] for r in rows])
    from infra.pg.mercator_erosion import _spearman
    rho_all = float(_spearman(x, y))
    per_type = {}
    for t in types:
        sel = [r for r in rows if r["type"] == t]
        if len(sel) >= 4:
            per_type[t] = float(_spearman([r["input_avoidance"] for r in sel],
                                          [r["row_avoidance"] for r in sel]))
    at.add("input_row_avoidance_spearman", value=rho_all, n=len(rows),
           flags={"per_type": per_type, "n_terms": n_terms,
                  "definition": ("rank correlation of input-side and row-side avoidance across "
                                 "cells; high means the row side is the input spectrum re-read")})
    # A HIGH correlation is the refuter. The threshold is stated rather than chosen after: two
    # statistics that are the same object under a transpose would track at |rho| near 1.
    SAME_OBJECT = 0.8
    distinct = bool(abs(rho_all) < SAME_OBJECT)
    # THE GATE NAME CARRIES THE TYPES IT WAS COMPUTED OVER. This job is invoked once per matrix
    # type (`--types q_proj,k_proj,v_proj,o_proj`, then `down_proj`, then `gate_proj`, then
    # `up_proj`), and it used to write ONE unscoped name every time. Several runs then banked the
    # same gate with different answers — attention ok=True (rho=-0.6425) and down_proj ok=False
    # (rho=+0.9773) — under a single prefix, where the report's latest-per-gate resolver has no
    # timestamp to separate them and returns whichever the iteration reached first. The arm's
    # verdict was decided by dict ordering (M-D105). A scoped name cannot collide, so per-type
    # answers stay per-type and an arm-level claim has to be assembled deliberately.
    scope = "_".join(t.replace("_proj", "") for t in types)
    gates.record(f"b1_row_side_is_distinct_from_input_side__{scope}", distinct,
                 spearman=rho_all, per_type=per_type, threshold=SAME_OBJECT, n_cells=len(rows),
                 types=list(types),
                 note=("M-B1's pre-logged direction, scoped to the types this run covered. A high "
                       "rank correlation across cells means the row side is the input spectrum "
                       "re-read; the two sides live in different spaces so their SUBSPACES cannot "
                       "be compared, but their avoidance numbers can"))
    # A PERMUTATION NULL, not the placeholder [0.0]. `surprise` takes its band from the SPREAD of
    # the control sample, and a one-element list has none — the arm crashed here with all 72 cells
    # measured and its gate already recorded (M-D79, the same crash M-D76 fixed in M-C1). The
    # matched null for "do these two columns co-vary across cells" is the same correlation with one
    # column shuffled: same values, same marginal distribution, pairing destroyed.
    prng = np.random.default_rng(a.seed)
    null_rhos = [float(_spearman(x, prng.permutation(y))) for _ in range(200)]
    at.add("input_row_avoidance_spearman_null", value=float(np.mean(null_rhos)), n=len(null_rhos),
           flags={"std": float(np.std(null_rhos, ddof=1)), "n_cells": len(rows),
                  "definition": ("mean of 200 cell-shuffled recomputations of the same rank "
                                 "correlation; the band the observed value is judged against")})
    sur.add(arm=ARM, expected_ref=expected_ref, cell="input_vs_row",
            statistic="input_row_avoidance_spearman",
            s=M.surprise(rho_all, float(np.mean(null_rhos)), null_rhos))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, "M-B1_distinct_result.json",
                 {"arm": ARM, "stage": "distinctness (input vs row)",
                  "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(),
                  "utc": M.utc_now(), "atlas": info, "base_repo": W.PG1_REPO,
                  "base_sha": base_sha, "types": list(types), "layers": layers,
                  "n_finetunes": n_terms, "rows": rows,
                  "spearman_input_vs_row": rho_all, "per_type_spearman": per_type,
                  "same_object_threshold": SAME_OBJECT, "distinct": distinct})
    print("B1D_OK " + json.dumps({"rows": info["n_rows"], "spearman": rho_all,
                                  "distinct": distinct, "per_type": per_type}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
