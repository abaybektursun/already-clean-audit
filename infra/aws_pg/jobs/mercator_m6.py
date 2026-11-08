"""M-6 — the rate-of-change checkpoint format (queue item 6). Wave P, Register B.

    THE FORMAT. A family of fine-tunes off one base is stored as three parts:

        stable block    the union of the per-layer cores, one orthonormal basis for the family
        vault           the base's action ON that block, kept at fp16 — exact where it matters
        bulk            everything else of the base, quantized to `bits`
        motion file     per fine-tune, a rank-r SVD of its delta

    The family parts are paid ONCE; only the motion file scales with the number of fine-tunes.

    REFUTER (pre-logged): the format costs more bytes than the own-SVD incumbent at matched
    capability.

THE INCUMBENT HAD TO BE REPLACED (defect M-D40). The obvious practitioner move — an fp16 backbone
plus a rank-r SVD of each delta — is not a fair comparison, because the format quantizes its
backbone to 4 bits and the fp16 incumbent does not. Against it the format is ~3x cheaper at one
fine-tune before the stable block does any work at all, so a gate on that ratio could not fail; it
was measuring 4-bit-vs-fp16 quantization wearing the format's name.

The incumbent that adjudicates is a UNIFORMLY QUANTIZED backbone at the SAME bits. Against that one
the format is strictly MORE expensive at equal rank — it adds the vault A and the shared basis on
top of an identical quantized bulk — so "is the format cheaper?" now cannot PASS, which is the same
defect pointing the other way.

So the comparison is at EQUAL BYTES: the format's surplus is spent giving the incumbent a higher
motion rank, and the incumbent is measured THERE. The format wins only if the vault's exact action
on the stable block is worth more than the extra rank the same bytes would have bought. That can
land either way, which is what makes it a measurement. The fp16 number is kept for context and is
not gated on.

CAPABILITY IS MEASURED AT EVERY RANK, not modelled. The curve is (bytes, capability) pairs where
capability is this job's own task readout on the decoded model, so "matched capability" is a
comparison of two measured points rather than of two error bounds.

TWO GATES BEFORE ANY BYTE IS COMPARED:
  * the round-trip equivalence gate, with tolerances frozen in this source — a format that does not
    reconstruct what it claims to store has no price worth discussing;
  * the capability of the DECODED model at the top of the rank ladder must be within a stated
    distance of the true fine-tune, or the ladder never reaches the thing being priced.

The instruments are `mercator_format`'s, imported: `union_stable_block`, `encode`/`decode`,
`roundtrip`, `assert_roundtrip`, `bytes_breakdown`, `capability_vs_bytes`. The fold machinery is
M-Q's, so the block lives in the same gauge the rest of the program uses.

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
from infra.pg import mercator_format as F
from infra.pg import mercator_program as P

ARM = "M-6"
# FROZEN BEFORE THE RUN. The round-trip tolerances the format must meet to be priced at all.
TOL_TOTAL = 5e-2
TOL_CORE = 1e-6
RANK_LADDER = (8, 16, 32, 64, 128)
BITS = 4
PPL_CORPUS = "wikitext2:window512"
TASK_CORPUS = "codeparrot-clean-valid:completion200"
# The decoded model at the top rank must land within this fraction of the true fine-tune's task
# loss, or the ladder never reaches what it is pricing.
MAX_TOP_RANK_TASK_GAP = 0.10


def main():
    ap = argparse.ArgumentParser(description="M-6 — the rate-of-change checkpoint format")
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="12,13,14,15,16,17")
    ap.add_argument("--ranks", default=",".join(str(r) for r in RANK_LADDER))
    ap.add_argument("--bits", type=int, default=BITS)
    ap.add_argument("--n-finetunes", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ranks = [int(r) for r in a.ranks.split(",") if r.strip()]
    M.check_dose_grid([float(r) for r in ranks], arm=ARM)
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()]
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_wp") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_6", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import gc
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoConfig, PaliGemmaProcessor
    from infra.pg import darwin_census as DC
    from infra.pg import darwin_train as T
    from infra.aws_pg.jobs.invariant_core_func import (_decoder_layers, _load_full_pg1,
                                                       _ppl_blankimg)
    from infra.aws_pg.jobs.invariant_core import _delete_cache
    from infra.aws_pg.jobs.darwin_w1_u import build_batches, eval_task_loss, text_task_rows

    assert torch.cuda.is_available(), "no CUDA"
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    mods = T.trainable_modules(_decoder_layers(model), types=types, layers=layers)
    keys = sorted(mods)
    pristine = {k: m.weight.detach().to(torch.float32).cpu().numpy().astype(np.float64)
                for k, m in mods.items()}
    _, eval_rows = text_task_rows(64, a.n_eval, a.seed)
    eval_b = build_batches(processor, eval_rows, "cuda")
    base_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
    base_task = eval_task_loss(model, eval_b)
    print(f"[M-6] base ppl {base_ppl:.5f} task {base_task:.5f} over {len(keys)} cells", flush=True)

    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    # THE STABLE BLOCK IS PER CELL, built from the family's per-layer cores for that matrix type,
    # because a matrix's input space is its own and a block spanning another type's coordinates
    # would not be a basis for it.
    block = {}
    for t in types:
        bases = [WT.load_bank(bank_dir, t, L, "core32") for L in layers]
        U = F.union_stable_block(bases)
        for L in layers:
            block[(t, L)] = U
        print(f"[M-6] stable block for {t}: rank {U.shape[1]} from {len(bases)} layer cores",
              flush=True)

    geom = DC.base_geometry(AutoConfig.from_pretrained(WT.PG1_REPO, revision=base_sha))
    accepted, log = DC.select_derivatives(api, WT.PG1_REPO, geom, n_accept=a.n_finetunes, out=out)
    shas = {r["id"]: api.model_info(r["id"]).sha for r in accepted}
    repos = [r["id"] for r in accepted]
    print(f"[M-6] family: {len(repos)} fine-tunes of {log['n_candidates']} candidates", flush=True)

    ft_mats = {}
    for repo in repos:
        m_ = M.load_decoder_matrices(repo, shas[repo], types, layers)
        ft_mats[repo] = {k: m_[k].to(torch.float32).numpy().astype(np.float64) for k in keys}
        del m_
        gc.collect()
        _delete_cache(repo)
    print(f"[M-6] deltas loaded for {len(ft_mats)} fine-tunes", flush=True)

    # ---- the round-trip gate, on the first fine-tune, before anything is priced ----------------
    probe = repos[0]
    rts = [F.roundtrip(pristine[k], ft_mats[probe][k], block[k], max(ranks), bits=a.bits)
           for k in keys]
    worst = max(rts, key=lambda r: r["rel_error_total"])
    gate = F.assert_roundtrip(worst, tol_total=TOL_TOTAL, tol_core=TOL_CORE)
    at.add("format_roundtrip_error", value=float(worst["rel_error_total"]), model_id=probe,
           flags={**gate, "tol_total": TOL_TOTAL, "tol_core": TOL_CORE, "bits": a.bits,
                  "worst_of_n_cells": len(rts),
                  "rel_error_base": worst["rel_error_base"],
                  "rel_error_delta": worst["rel_error_delta"]})
    print(f"[M-6] round-trip gate passed: worst rel_error_total "
          f"{worst['rel_error_total']:.3e}, core action {worst['core_action_error']:.3e}",
          flush=True)

    # ---- capability at every rank, MEASURED --------------------------------------------------
    def install(W_by_key):
        with torch.no_grad():
            for k, m in mods.items():
                m.weight.data.copy_(torch.as_tensor(W_by_key[k], device="cuda",
                                                    dtype=m.weight.dtype))

    true_task, true_ppl = {}, {}
    for repo in repos:
        install(ft_mats[repo])
        model.eval()
        true_task[repo] = eval_task_loss(model, eval_b)
        true_ppl[repo] = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
    print(f"[M-6] true fine-tune task losses {min(true_task.values()):.4f}-"
          f"{max(true_task.values()):.4f}", flush=True)

    # THE MATCHED-BITS INCUMBENT (M-D40). Uniform quantization of the same backbone at the same
    # `bits`, carrying the SAME motion file. It is CHEAPER than the format at every rank, so it is
    # the incumbent that can refute the arm; the fp16 backbone the format used to be priced against
    # makes the format look ~3x cheaper before the stable block does any work at all. Quantizing
    # does not depend on r, so it is done once.
    inc_base = {k: CEQ.quantize_uniform(pristine[k], bits=a.bits) for k in keys}
    m0, n0 = pristine[keys[0]].shape
    kb = int(block[keys[0]].shape[1])
    # The rank the incumbent buys with the format's own bytes — the equal-bytes comparison. At the
    # format's OWN rank the incumbent is simply cheaper, so that comparison can only ever go one
    # way and measures nothing.
    aff = {r: F.affordable_incumbent_rank(r, m=m0, n=n0, k_block=kb, n_finetunes=len(repos),
                                          bits=a.bits, n_matrices_sharing_block=len(layers))
           for r in ranks}
    print(f"[M-6] byte-matched incumbent ranks: {aff}", flush=True)

    cap, inc_cap = {}, {}
    rows = []
    for r in ranks:
        per_repo = {}
        rp = aff[r]
        for repo in repos:
            enc = {k: F.encode(pristine[k], ft_mats[repo][k], block[k], r, bits=a.bits)
                   for k in keys}
            install({k: F.decode(enc[k]) for k in keys})
            model.eval()
            task = eval_task_loss(model, eval_b)
            ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=a.n_seq)["ppl"])
            # The incumbent at the rank those same bytes afford: plain quantized backbone, RICHER
            # motion. The format wins only if the vault beats the extra rank it gave up to pay for
            # itself.
            enc_i = {k: F.encode(pristine[k], ft_mats[repo][k], block[k], rp, bits=a.bits)
                     for k in keys}
            install({k: inc_base[k] + enc_i[k]["motion"]["U"] @ enc_i[k]["motion"]["Vt"]
                     for k in keys})
            model.eval()
            inc_task = eval_task_loss(model, eval_b)
            per_repo[repo] = {"task_loss": task, "ppl": ppl,
                              "task_gap": task - true_task[repo],
                              "incumbent_rank": rp, "incumbent_task_loss": inc_task}
            at.add("format_capability", value=task, dose=float(r), corpus_id=TASK_CORPUS,
                   model_id=repo, model_sha=shas[repo],
                   flags={"rank": r, "bits": a.bits, "ppl": ppl,
                          "true_task_loss": true_task[repo], "true_ppl": true_ppl[repo],
                          "task_gap": task - true_task[repo], "base_task_loss": base_task,
                          "scheme": "format_vault"})
            at.add("format_capability", value=inc_task, dose=float(rp), corpus_id=TASK_CORPUS,
                   model_id=repo, model_sha=shas[repo],
                   incumbent_id="uniform_quantized_backbone",
                   flags={"rank": rp, "format_rank_matched": r, "bits": a.bits,
                          "scheme": "incumbent_byte_matched"})
        cap[r] = float(np.mean([v["task_loss"] for v in per_repo.values()]))
        inc_cap[rp] = float(np.mean([v["incumbent_task_loss"] for v in per_repo.values()]))
        rows.append({"rank": r, "mean_task_loss": cap[r], "per_finetune": per_repo,
                     "incumbent_rank": rp, "mean_incumbent_task_loss": inc_cap[rp],
                     "mean_task_gap": float(np.mean([v["task_gap"] for v in per_repo.values()]))})
        print(f"[M-6] rank {r:4d}: mean task {cap[r]:.5f} "
              f"incumbent(r={rp}) {inc_cap[rp]:.5f} "
              f"(gap to true {rows[-1]['mean_task_gap']:+.5f})", flush=True)
        M.s3_sync_incremental(out)

    # THE LADDER MUST REACH THE THING IT PRICES. If even the top rank is far from the true
    # fine-tune, every "matched capability" comparison below is between two points neither of
    # which is the model anyone would ship.
    top_gap = rows[-1]["mean_task_gap"] / max(abs(base_task - np.mean(list(true_task.values()))),
                                              1e-9)
    reaches = bool(abs(top_gap) <= MAX_TOP_RANK_TASK_GAP)
    gates.record("m6_ladder_reaches_the_true_finetune", reaches,
                 top_rank=ranks[-1], mean_task_gap=rows[-1]["mean_task_gap"],
                 relative_gap=float(top_gap), max_allowed=MAX_TOP_RANK_TASK_GAP,
                 outcome=None if reaches else M.VOID_INSTRUMENT,
                 note=("the gap is measured against the base-to-finetune task distance, so it is "
                       "the fraction of the fine-tune's own effect that the format loses"))

    m, n, k_block = m0, n0, kb
    # higher_is_better=False: this arm's readout is a task LOSS, so `capability_gain` has to be
    # signed the other way or every comparison below reads backwards.
    curve = F.capability_vs_bytes(ranks, cap, inc_cap, m=m, n=n, k_block=k_block,
                                  n_finetunes=len(repos), bits=a.bits,
                                  n_matrices_sharing_block=len(layers),
                                  higher_is_better=False)
    n_weights = m * n * len(repos)
    for row in curve:
        at.add("format_bytes_per_weight", value=float(row["total_bits"] / n_weights),
               dose=float(row["r"]),
               flags={**row, "n_matrices_sharing_block": len(layers),
                      "incumbent_bits_per_weight": row["incumbent_total_bits"] / n_weights,
                      "definition": ("total shipped bits divided by the weights delivered, i.e. "
                                     "one matrix times the family size")})
    # A WIN IS BOTH HALVES. The two schemes ship the identical motion file, so the motion side IS
    # matched by construction — but the BACKBONE side is not, and that is where all the bytes are.
    # Against a backbone quantized to the same bits the format is strictly more expensive, so the
    # arm can only win on capability at no-more-bytes, and both halves are measured.
    wins = F.pareto_wins(curve)
    gates.record("m6_format_is_cheaper_at_matched_capability", bool(wins),
                 pareto_win_ranks=wins,
                 total_bits={row["r"]: row["total_bits"] for row in curve},
                 incumbent_bits={row["r"]: row["incumbent_total_bits"] for row in curve},
                 capability_gain={row["r"]: row["capability_gain"] for row in curve},
                 bits_ratio_vs_fp16={row["r"]: row["bits_ratio_vs_fp16"] for row in curve},
                 n_finetunes=len(repos), k_block=k_block,
                 n_matrices_sharing_block=len(layers),
                 note=("the incumbent is a uniformly-quantized backbone at the SAME bits carrying "
                       "the SAME motion file. The old fp16-backbone incumbent is kept for context "
                       "only: against it the format is ~3x cheaper at one fine-tune before the "
                       "stable block does anything, so that gate could not fail (defect M-D40)"))
    sur.add(arm=ARM, expected_ref=expected_ref, cell=f"r={ranks[-1]}",
            statistic="format_bytes_per_weight",
            s=M.surprise(curve[-1]["total_bits"], curve[-1]["incumbent_total_bits"],
                         [row["incumbent_total_bits"] for row in curve]))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "base_ppl": base_ppl, "base_task_loss": base_task, "types": list(types),
                  "layers": layers, "ranks": ranks, "bits": a.bits, "k_block": k_block,
                  "family": repos, "true_task_loss": true_task, "true_ppl": true_ppl,
                  "tol_total": TOL_TOTAL, "tol_core": TOL_CORE,
                  "roundtrip_gate": gate, "capability_rows": rows, "curve": curve,
                  "pareto_win_ranks": wins, "byte_matched_incumbent_ranks": aff,
                  "ladder_reaches_true_finetune": reaches})
    print("M6_OK " + json.dumps({"rows": info["n_rows"], "k_block": k_block,
                                 "pareto_wins": wins,
                                 "reaches": reaches}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
