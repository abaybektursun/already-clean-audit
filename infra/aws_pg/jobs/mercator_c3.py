"""M-C3 — progressive delivery: capability against bytes on the r-ladder (queue C3). Wave P.

    THE DELIVERABLE. A fine-tune shipped as a STREAM: the first bytes give a usable model, each
    further chunk improves it, and the receiver can stop anywhere. The r-ladder format is that
    stream — rank-r motion on top of a family-level base — and the question is whether it delivers
    more capability per byte than the obvious alternative.

    REFUTER (pre-logged): capability-vs-bytes no better than the own-SVD incumbent.

TWO FINE-TUNES, NOT ONE, and the second is the point. A single model's curve cannot distinguish
"this format is good" from "this fine-tune happens to be low-rank", so the arm runs the ladder on
`ft-vqav2` and on a second fine-tune with a valid readout, and reports both curves.

CAPABILITY IS THE FINE-TUNE'S OWN TASK, not a proxy. A streaming format for a VQA model is worth
something if the partially-delivered model does VQA; wikitext perplexity would measure whether the
stream damaged the backbone, which is a different and easier question. Perplexity is reported
alongside for exactly that reason.

THE READOUT-VALIDITY GATE COMES FIRST (§0.2.9): each fine-tune must beat the BASE on its own task,
or "capability delivered" has no scale. A fine-tune whose task readout does not separate it from
the base is recorded VOID-readout and its curve is not interpreted.

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

ARM = "M-C3"
RANK_LADDER = (4, 8, 16, 32, 64, 128)
BITS = 4
# The two fine-tunes and the battery task each is measured on. Named here so the pairing of a
# model with its own task is auditable rather than inferred from a substring.
DELIVERIES = (("google/paligemma-3b-ft-vqav2-224", "vqav2"),
              ("google/paligemma-3b-ft-cococap-224", "coco_captions"))


def main():
    ap = argparse.ArgumentParser(description="M-C3 — progressive delivery on the r-ladder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="q_proj,o_proj")
    ap.add_argument("--layers", default="")
    ap.add_argument("--ranks", default=",".join(str(r) for r in RANK_LADDER))
    ap.add_argument("--bits", type=int, default=BITS)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    H.hf_setup()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ranks = [int(r) for r in a.ranks.split(",") if r.strip()]
    M.check_dose_grid([float(r) for r in ranks], arm=ARM)
    types = tuple(t.strip() for t in a.types.split(",") if t.strip())
    layers = [int(x) for x in a.layers.split(",") if x.strip()] or list(range(H.N_LAYERS))
    meta = P.ARMS[ARM]
    exp = M.fetch_expectations(H.cache_dir(out, "mercator_wp") / "expectations.jsonl")
    expected_ref = exp.require(ARM)
    at = M.Atlas(out, arm=ARM, run_id="m_c3", register=meta["register"], wave=meta["wave"],
                 expected_ref=expected_ref, queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    import gc
    import torch
    from huggingface_hub import HfApi
    from transformers import PaliGemmaProcessor
    from infra.pg import darwin_train as T
    from infra.pg import projector_ablation as PA
    from infra.aws_pg.jobs.invariant_core_func import (_decoder_layers, _load_full_pg1,
                                                       _ppl_blankimg)
    from infra.aws_pg.jobs.invariant_core import _delete_cache

    assert torch.cuda.is_available(), "no CUDA"
    # The corpus (CIDEr-D) task needs pycocoevalcap, which the AMI does not carry. Installed here,
    # BEFORE any scoring, so the failure mode is a loud install error at second zero rather than a
    # crash after the first task has already been evaluated.
    M.ensure_pycocoevalcap()
    api = HfApi()
    base_sha = api.model_info(WT.PG1_REPO).sha
    model = _load_full_pg1(base_sha, "cuda")
    processor = PaliGemmaProcessor.from_pretrained(WT.PG1_REPO, revision=base_sha)
    mods = T.trainable_modules(_decoder_layers(model), types=types, layers=layers)
    keys = sorted(mods)
    pristine = {k: m.weight.detach().to(torch.float32).cpu().numpy().astype(np.float64)
                for k, m in mods.items()}
    bank_dir = M.fetch_all_banks(H.cache_dir(out, "banks"))
    block = {}
    for t in types:
        U = F.union_stable_block([WT.load_bank(bank_dir, t, L, "core32") for L in layers])
        for L in layers:
            block[(t, L)] = U
        print(f"[M-C3] stable block for {t}: rank {U.shape[1]}", flush=True)

    tasks = {t.name: t for t in PA.build_tasks()}
    wanted = {name for _r, name in DELIVERIES}
    assert wanted <= set(tasks), f"battery has {sorted(tasks)}, need {sorted(wanted)}"
    rows_by_task = {name: PA.load_eval_items(tasks[name], a.n_eval, seed=a.seed)
                    for name in wanted}

    def install(W_by_key):
        with torch.no_grad():
            for k, m in mods.items():
                m.weight.data.copy_(torch.as_tensor(W_by_key[k], device="cuda",
                                                    dtype=m.weight.dtype))

    def score(name):
        return PA.eval_task(model, processor, tasks[name], rows_by_task[name], "cuda",
                            batch_size=8)

    base_scores = {name: score(name) for name in wanted}
    base_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=16)["ppl"])
    for name, s in base_scores.items():
        print(f"[M-C3] base on {name}: {s['score']:.4f} CI={s['ci95']}", flush=True)

    results = {}
    for repo, task_name in DELIVERIES:
        sha = api.model_info(repo).sha
        mats = M.load_decoder_matrices(repo, sha, types, layers)
        ft = {k: mats[k].to(torch.float32).numpy().astype(np.float64) for k in keys}
        del mats
        gc.collect()
        _delete_cache(repo)
        install(ft)
        model.eval()
        true_s = score(task_name)
        true_ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=16)["ppl"])
        # §0.2.9: the delivery has no scale if the fine-tune does not beat the base on its own task.
        headroom = float(true_s["score"] - base_scores[task_name]["score"])
        valid = bool(headroom > 0 and
                     true_s["ci95"][0] > base_scores[task_name]["ci95"][1])
        gates.record(f"c3_readout_valid__{task_name}", valid,
                     base=base_scores[task_name]["score"], finetune=true_s["score"],
                     base_ci=base_scores[task_name]["ci95"], ft_ci=true_s["ci95"],
                     headroom=headroom, outcome=None if valid else M.VOID_READOUT,
                     note=("the fine-tune must beat the base on its OWN task with separated "
                           "intervals, or 'fraction of capability delivered' has no denominator"))
        at.add("delivery_capability", value=float(true_s["score"]), dose=float(max(ranks) * 4),
               corpus_id=tasks[task_name].dataset, model_id=repo, model_sha=sha,
               ci_lo=true_s["ci95"][0], ci_hi=true_s["ci95"][1], n=true_s["n"],
               incumbent_id="full_finetune",
               flags={"role": "the true fine-tune (full delivery)", "task": task_name,
                      "ppl": true_ppl, "base_score": base_scores[task_name]["score"],
                      "headroom": headroom, "readout_valid": valid})

        # THE MATCHED-BITS INCUMBENT (M-D40). A plain uniformly-quantized backbone at the SAME
        # `bits`, carrying the SAME rank-r motion file. It is CHEAPER than the format at every rank
        # — the format adds A and the shared basis on top of an identical quantized bulk — so it is
        # the incumbent that can actually refute the arm, and its capability has to be measured
        # rather than assumed. Quantizing the backbone does not depend on r, so it is done once.
        inc_base = {k: CEQ.quantize_uniform(pristine[k], bits=a.bits) for k in keys}
        m0, n0 = pristine[keys[0]].shape
        kb = int(block[keys[0]].shape[1])
        # The rank the incumbent buys with the format's own bytes. Measuring it at the FORMAT's
        # rank would compare two schemes at two prices; this is the equal-bytes comparison.
        aff = {r: F.affordable_incumbent_rank(r, m=m0, n=n0, k_block=kb, n_finetunes=1,
                                              bits=a.bits, n_matrices_sharing_block=len(layers))
               for r in ranks}
        print(f"[M-C3] byte-matched incumbent ranks: {aff}", flush=True)

        cap, inc_cap, per_rank = {}, {}, []
        for r in ranks:
            enc = {k: F.encode(pristine[k], ft[k], block[k], r, bits=a.bits) for k in keys}
            dec = {k: F.decode(enc[k]) for k in keys}
            install(dec)
            model.eval()
            s = score(task_name)
            ppl = float(_ppl_blankimg(model, processor, "cuda", n_seq=16)["ppl"])
            frac = ((s["score"] - base_scores[task_name]["score"]) / headroom
                    if headroom > 0 else float("nan"))
            cap[r] = float(s["score"])
            # The incumbent at the rank the same bytes afford: plain quantized backbone, a RICHER
            # motion file. The format wins only if the vault's exact action on the stable block
            # beats the extra rank those bytes would otherwise have bought.
            rp = aff[r]
            enc_i = {k: F.encode(pristine[k], ft[k], block[k], rp, bits=a.bits) for k in keys}
            install({k: inc_base[k] + enc_i[k]["motion"]["U"] @ enc_i[k]["motion"]["Vt"]
                     for k in keys})
            model.eval()
            si = score(task_name)
            inc_cap[rp] = float(si["score"])
            per_rank.append({"rank": r, "score": float(s["score"]), "ci95": s["ci95"],
                             "ppl": ppl, "fraction_delivered": frac, "incumbent_rank": rp,
                             "incumbent_score": float(si["score"]), "incumbent_ci95": si["ci95"]})
            at.add("delivery_capability", value=float(s["score"]), dose=float(r),
                   corpus_id=tasks[task_name].dataset, model_id=repo, model_sha=sha,
                   ci_lo=s["ci95"][0], ci_hi=s["ci95"][1], n=s["n"],
                   flags={"rank": r, "bits": a.bits, "task": task_name, "ppl": ppl,
                          "base_ppl": base_ppl, "fraction_delivered": frac,
                          "true_score": float(true_s["score"]), "scheme": "format_vault"})
            at.add("delivery_capability", value=float(si["score"]), dose=float(rp),
                   corpus_id=tasks[task_name].dataset, model_id=repo, model_sha=sha,
                   ci_lo=si["ci95"][0], ci_hi=si["ci95"][1], n=si["n"],
                   incumbent_id="uniform_quantized_backbone",
                   flags={"rank": rp, "format_rank_matched": r, "bits": a.bits,
                          "task": task_name, "scheme": "incumbent_byte_matched"})
            print(f"[M-C3] {repo.split('/')[-1]:26s} r={r:4d} {task_name}={s['score']:.4f} "
                  f"incumbent(r={rp})={si['score']:.4f} delivered={frac:+.3f} ppl={ppl:.3f}",
                  flush=True)
            M.s3_sync_incremental(out)

        curve = F.capability_vs_bytes(ranks, cap, inc_cap, m=m0, n=n0, k_block=kb, n_finetunes=1,
                                      bits=a.bits, n_matrices_sharing_block=len(layers))
        for row in curve:
            at.add("delivery_bytes", value=float(row["total_bits"]), dose=float(row["r"]),
                   model_id=repo, model_sha=sha,
                   flags={**row, "task": task_name,
                          "incumbent": ("uniform-quantized backbone at the same bits + the same "
                                        "rank-r motion file")})
        wins = F.pareto_wins(curve)
        results[repo] = {"task": task_name, "sha": sha, "true_score": float(true_s["score"]),
                         "true_ppl": true_ppl, "base_score": base_scores[task_name]["score"],
                         "headroom": headroom, "readout_valid": valid,
                         "per_rank": per_rank, "curve": curve, "pareto_wins": wins}
        gates.record(f"c3_beats_own_svd_incumbent__{task_name}", bool(wins),
                     pareto_win_ranks=wins,
                     bits_ratio={row["r"]: row["bits_ratio"] for row in curve},
                     capability_gain={row["r"]: row["capability_gain"] for row in curve},
                     bits_ratio_vs_fp16={row["r"]: row["bits_ratio_vs_fp16"] for row in curve},
                     note=("a win is BOTH no-more-bytes AND more-capability against a backbone "
                           "quantized to the same bits carrying the identical motion file. The "
                           "format is strictly MORE expensive than that incumbent, so the only "
                           "way it wins is the vault's exact action on the stable block — which "
                           "is its actual claim. The fp16-backbone ratio is recorded for context "
                           "and is NOT gated on: it is 4-bit-vs-fp16 quantization wearing the "
                           "format's name and cannot fail (defect M-D40)"))
        del ft
        gc.collect()

    at.check_readout_corpus(["delivery_capability"])
    valid_tasks = [r["task"] for r in results.values() if r["readout_valid"]]
    gates.record("c3_at_least_one_valid_delivery", bool(valid_tasks), valid_tasks=valid_tasks,
                 outcome=None if valid_tasks else M.VOID_READOUT)
    first = next(iter(results.values()))
    sur.add(arm=ARM, expected_ref=expected_ref, cell=f"r={ranks[-1]}",
            statistic="delivery_bytes",
            s=M.surprise(first["curve"][-1]["total_bits"],
                         first["curve"][-1]["incumbent_total_bits"],
                         [row["incumbent_total_bits"] for row in first["curve"]]))

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{ARM}_result.json",
                 {"arm": ARM, "queue_item": meta["queue_item"], "refuter": meta["refuter"],
                  "expected_ref": expected_ref, "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "base_repo": WT.PG1_REPO, "base_sha": base_sha,
                  "base_ppl": base_ppl,
                  "base_scores": {k: float(v["score"]) for k, v in base_scores.items()},
                  "types": list(types), "layers": layers, "ranks": ranks, "bits": a.bits,
                  "deliveries": {r: t for r, t in DELIVERIES}, "results": results,
                  "valid_tasks": valid_tasks})
    print("C3_OK " + json.dumps({"rows": info["n_rows"], "valid_tasks": valid_tasks,
                                 "pareto_wins": {r: v["pareto_wins"]
                                             for r, v in results.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
