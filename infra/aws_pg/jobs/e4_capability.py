"""E4 capability readout — GraftLM capability-readout prereg
(research/2026-05-30_graftlm_capability_readout_prereg.md).

Turns E4's behavioural KL-closure (R_KL) into a CAPABILITY claim: does the no-retrain rank-k SVD edit
of the base->pt modality-install delta recover zero-shot TASK CAPABILITY in proportion to the KL gap
it closes, less (cliff / "cheap talk"), or more (lead)?

Two modes:

  --mode preflight   The runbook's fail-fast gate (prereg §9 step 1). Native zero-shot
                     generate-and-score of the two anchors on RefCOCO at small N; decides whether the
                     native-capability axis has dynamic range BEFORE freeze.
                       CEILING = PG2-pt (full install)                 = build_pair('base-pt').pg_after
                       FLOOR   = PG2 vision/proj + base gemma-2-2b dec  = build_pair('base-pt').pg_before
                     Gate: CEILING-FLOOR >= 0.15 (frozen margin). [RAN 2026-05-30: PASS, 0.28 vs 0.00,
                     padding artifact ruled out by a batch=1 rerun — 0.28 is genuine pt-224 zero-shot.]

  --mode sweep       The frozen confirmatory sweep (prereg §9 step 3). For each task, build the
                     no-retrain rank-k SVD edit family of the base->pt LM-decoder delta (BIT-IDENTICAL
                     to E4 e4_compress_curve.py: same truncated SVD, _set_decoder, anchors, nulls) and
                     native zero-shot generate-and-score each config:
                       configs = {FLOOR, rank-8/32/128/512/1024, CEILING=PG2-pt, E_scale, N3@8, N3@128}
                       tasks   = {refcoco (acc@0.5), textvqa (ANLS), coco (CIDEr-D)}
                     Capability-recovery fraction R_cap(k) = (m(cfg)-m(FLOOR))/(m(CEILING)-m(FLOOR)),
                     the exact parallel of E4's R_KL(k). The headline contrast is R_cap(k) vs E4's
                     R_KL(k) (recomputed from s3://.../results/e4_basept_v2 at analysis time).

Metric + prompts reused VERBATIM from infra.pg.battery_tasks (NOT its transfer fine-tune recipe — a
trained bridge would reintroduce the retrained-bridge S_B estimand collision E4's native R_KL
avoids). Generation is greedy; forward shape is the battery_tasks eval path verbatim.

GPU job. hf_online_setup() first (gated weights + streamed eval sets). No try/except around the
science — a wrong assumption RAISES with a full trace; a no-dynamic-range outcome is RECORDED.

    # preflight (done):
    pg run infra.aws_pg.jobs.e4_capability --args="--mode=preflight --n=100" --type p4d.24xlarge \
        --s3-prefix results/e4_capreadout_preflight --budget-min 45
    # frozen sweep:
    pg run infra.aws_pg.jobs.e4_capability --args="--mode=sweep --n=2000" \
        --type g6e.8xlarge,p4d.24xlarge --s3-prefix results/e4_capreadout_sweep --budget-min 300
"""
from __future__ import annotations

import argparse
import json
import pathlib

from infra.aws_pg.jobs.e4_compress_curve import _decoder_keys, _set_decoder

SEED = 1337                 # prereg manifest selection + bootstrap seed
K_SWEEP = (8, 32, 128, 512, 1024)   # rank-k per decoder matrix (E4-identical)
K_MAX = 1024                # truncated-SVD budget (just under the ~1440 eff-rank)
N3_K = 128                  # mid-rank random-null reference
GEN_BATCH = 8               # greedy generation micro-batch (battery_tasks eval shape; batch-invariant
                            # — verified by the preflight batch=1 == batch=8 check)
N_BOOT = 2000               # bootstrap draws (per-item tasks)
N_BOOT_CORPUS = 300         # bootstrap draws for the corpus metric (CIDEr; tokenize-once)
N_PREFLIGHT = 100           # preflight anchor N
N_SWEEP = 2000              # frozen sweep N/task (preflight-informed: CEILING~0.28 -> N=2000 makes the
                            # single-task R_cap CI < the frozen margin delta=0.10; see journal)
GATE_MIN_RANGE = 0.15       # frozen dynamic-range margin (CEILING - FLOOR)
GATE_FLOOR_MAX = 0.05       # FLOOR ~ 0 (a no-install decoder cannot localise)
# coco_captions DEFERRED from the frozen run: pycocoevalcap (+java for PTBTokenizer) is not on the
# pg AMI (smoke 2026-05-30 ModuleNotFoundError). RefCOCO (MDE) + TextVQA both have dynamic range
# (denom 0.31 / 0.20) and carry the verdict. COCO is runnable via --tasks=coco_captions once the dep
# is baked into the AMI; its CIDEr path (tokenize-once bootstrap) is implemented + smoke-pending.
TASKS_ALL = ("refcoco", "textvqa")


# ===========================================================================
# Eval-set selection + generation/scoring (battery_tasks prompts + scorers; NOT the transfer recipe).
# ===========================================================================

def _eval_rows(task, n, seed):
    """First-n of a fixed-seed (1337) shuffle of the task's eval split — deterministic, recorded,
    matches the prereg manifest selection. Full (non-streaming) load so .shuffle(seed).select is
    reproducible (battery_tasks._load_train_eval pattern)."""
    from datasets import load_dataset
    ds = load_dataset(task.dataset, split=task.eval_split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [ds[i] for i in range(len(ds))]


def _generate(model, proc, task, rows, device, batch):
    """Native zero-shot greedy generation; returns predictions aligned to `rows`. Forward shape is
    VERBATIM the battery_tasks eval path (proc -> generate do_sample=False -> decode the new tokens).
    RefCOCO keeps special tokens so the <locDDDD> box decodes."""
    import torch
    keep_special = task.name == "refcoco"
    preds = []
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        images = [task.image_fn(r).convert("RGB") for r in chunk]
        prompts = [task.prompt_fn(r) for r in chunk]
        inp = proc(text=prompts, images=images, return_tensors="pt", padding="longest").to(device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        in_len = inp["input_ids"].shape[-1]
        with torch.inference_mode():
            gen = model.generate(**inp, max_new_tokens=task.max_new_tokens, do_sample=False)
        decoded = proc.batch_decode(gen[:, in_len:], skip_special_tokens=not keep_special)
        preds.extend(p.strip() for p in decoded)
    return preds


def _is_corpus(task):
    from infra.pg import battery_tasks as B
    return task.name in B.CORPUS_METRIC_TASKS


def _per_item_scores(task, rows, preds):
    """np.array of per-example scores (acc@0.5 / ANLS). Per-item tasks only."""
    import numpy as np
    return np.array([task.score_fn(p, r) for r, p in zip(rows, preds)], dtype=float)


def _corpus_tokenized(rows, preds):
    """PTB-tokenize the candidate/reference corpus ONCE (CIDEr); return (tok_res, tok_gts) dicts so
    the bootstrap resamples pre-tokenized strings (no per-draw java spawn)."""
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from infra.pg.battery_tasks import _coco_caps
    cand = {i: [{"caption": preds[i]}] for i in range(len(rows))}
    ref = {i: [{"caption": c} for c in _coco_caps(rows[i])] for i in range(len(rows))}
    tok = PTBTokenizer()
    return tok.tokenize(cand), tok.tokenize(ref)


def _cider_from_tokenized(tok_res, tok_gts, ids=None):
    """CIDEr-D over a (sub)set of pre-tokenized ids. When `ids` is a resample (with replacement), the
    draw is REKEYED by position (j: id) so duplicate examples are preserved — a {id: ...} dict
    comprehension would silently dedup them and break the bootstrap."""
    from pycocoevalcap.cider.cider import Cider
    if ids is None:
        res, gts = tok_res, tok_gts
    else:
        res = {j: tok_res[i] for j, i in enumerate(ids)}
        gts = {j: tok_gts[i] for j, i in enumerate(ids)}
    return float(Cider().compute_score(gts, res)[0])


def _boot_ci(scores, n_boot, seed):
    """Bootstrap mean CI for a per-item score array (also used by the preflight)."""
    import numpy as np
    a = np.asarray(scores, dtype=float)
    assert np.isfinite(a).all(), f"non-finite scores (harness bug): {a[~np.isfinite(a)]}"
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(a, a.size, replace=True).mean() for _ in range(n_boot)])
    return {"mean": float(a.mean()), "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)), "n": int(a.size)}


# ===========================================================================
# PREFLIGHT (prereg §9 step 1) — the dynamic-range gate.
# ===========================================================================

def run_preflight(out, n, batch, device):
    import json as _json
    from infra.pg import battery_tasks as B
    from infra.pg import diff_harvest as H

    task = B.REFCOCO_TASK
    rows = _eval_rows(task, n, SEED)
    print(f"[capreadout/preflight] task={task.name} split={task.eval_split} n={len(rows)} "
          f"metric={task.metric_name} max_new_tokens={task.max_new_tokens}", flush=True)

    pg_after, pg_before, proc, _paths, _img, gate = H.build_pair("base-pt", device)
    print(f"[capreadout/preflight] swap_gate={_json.dumps(gate, default=str)}", flush=True)

    ceil_ci = _boot_ci(_per_item_scores(task, rows, _generate(pg_after, proc, task, rows, device, batch)),
                       N_BOOT, SEED)
    print(f"[capreadout/preflight] CEILING (PG2-pt) acc={ceil_ci['mean']:.4f} "
          f"[{ceil_ci['lo']:.4f},{ceil_ci['hi']:.4f}]", flush=True)
    floor_ci = _boot_ci(_per_item_scores(task, rows, _generate(pg_before, proc, task, rows, device, batch)),
                        N_BOOT, SEED)
    print(f"[capreadout/preflight] FLOOR (base decoder) acc={floor_ci['mean']:.4f} "
          f"[{floor_ci['lo']:.4f},{floor_ci['hi']:.4f}]", flush=True)

    dyn = ceil_ci["mean"] - floor_ci["mean"]
    gate_pass = bool(ceil_ci["mean"] >= GATE_MIN_RANGE and dyn >= GATE_MIN_RANGE)
    res = {"experiment": "e4_capability_preflight", "stage": "preflight", "task": task.name,
           "n": len(rows), "seed": SEED, "ceiling": ceil_ci, "floor": floor_ci, "dynamic_range": dyn,
           "gate": {"min_range": GATE_MIN_RANGE, "pass": gate_pass}, "swap_gate": gate}
    (out / "e4_capability_preflight.json").write_text(json.dumps(res, indent=2))
    print("E4_CAP_PREFLIGHT_OK", json.dumps({"ceiling": ceil_ci["mean"], "floor": floor_ci["mean"],
                                             "dynamic_range": dyn, "gate_pass": gate_pass}), flush=True)


# ===========================================================================
# SWEEP (prereg §9 step 3) — the frozen confirmatory R_cap(k) curve.
# ===========================================================================

def _build_edit_family(pg_after, pg_before, paths, device):
    """E4-identical no-retrain SVD edit family of the base->pt LM-decoder delta. Returns
    (lm_edit, base_sd, after_sd, dkeys, builders) where builders maps a config name -> (edited_decoder
    overrides, base_override state-dict). FLOOR/rank-k/E_scale/N3 hold per-layer NORMS AT BASE (only
    the 2-D decoder matrices change); CEILING = the FULL PG2-pt LM (norms included)."""
    import torch
    lm_after = pg_after.get_submodule(paths["language_model"])
    lm_edit = pg_before.get_submodule(paths["language_model"])
    base_sd = {k: v.detach().clone().cpu() for k, v in lm_edit.state_dict().items()}
    after_sd = {k: v.detach().clone().cpu() for k, v in lm_after.state_dict().items()}
    dkeys = _decoder_keys()
    for k in dkeys:
        assert k in base_sd and k in after_sd, f"missing decoder key {k}"

    print(f"[capreadout/sweep] truncated SVD (q={K_MAX}+256) of {len(dkeys)} decoder dW ...", flush=True)
    svd, dW_frob = {}, {}
    for k in dkeys:
        dW = (after_sd[k].float() - base_sd[k].float()).to(device)
        q = min(K_MAX + 256, min(dW.shape))
        U, S, V = torch.svd_lowrank(dW, q=q, niter=8)
        svd[k] = (U[:, :K_MAX].cpu(), S[:K_MAX].cpu(), V[:, :K_MAX].cpu())
        dW_frob[k] = float(dW.norm().item())
        del dW, U, S, V
        torch.cuda.empty_cache()

    def _rankk(k):
        d = {}
        for key in dkeys:
            U, S, V = svd[key]
            kk = min(k, S.shape[0])
            dWk = (U[:, :kk].to(device) * S[:kk].to(device)) @ V[:, :kk].to(device).T
            d[key] = (base_sd[key].float().to(device) + dWk).to(torch.bfloat16)
        return d

    def _escale():
        d = {}
        for key in dkeys:
            nb = base_sd[key].float().norm().clamp_min(1e-8)
            na = after_sd[key].float().norm()
            d[key] = (base_sd[key].float() * (na / nb)).to(torch.bfloat16)
        return d

    def _n3(rank):
        g = torch.Generator(device="cpu").manual_seed(SEED)
        d = {}
        for key in dkeys:
            out_d, in_d = base_sd[key].shape
            Rm = torch.randn(out_d, in_d, generator=g)
            Rm = Rm * (dW_frob[key] / Rm.norm().clamp_min(1e-8))
            U, S, V = torch.svd_lowrank(Rm.to(device), q=min(rank + 16, min(out_d, in_d)), niter=4)
            kk = min(rank, S.shape[0])
            d[key] = (base_sd[key].float().to(device) + (U[:, :kk] * S[:kk]) @ V[:, :kk].T).to(torch.bfloat16)
            del Rm, U, S, V
        return d

    def _ldesc(k):
        return int(sum(min(k, svd[key][1].shape[0]) * (base_sd[key].shape[0] + base_sd[key].shape[1])
                       for key in dkeys))

    # (name, edited-decoder overrides, base_override) — applied via _set_decoder(lm_edit, base_override, edited).
    builders = [("FLOOR", {}, base_sd)]
    for k in K_SWEEP:
        builders.append((f"rank{k}", _rankk(k), base_sd))
    builders.append(("CEILING", {}, after_sd))                       # full PG2-pt LM (norms included)
    builders.append(("E_scale", _escale(), base_sd))                 # norm-only magnitude null
    for r in sorted({min(K_SWEEP), N3_K}):
        builders.append((f"N3_rank{r}", _n3(r), base_sd))            # random rank-r matched to ||dW||_F
    edited_params = int(sum(base_sd[k].shape[0] * base_sd[k].shape[1] for k in dkeys))
    ldesc = {f"rank{k}": _ldesc(k) for k in K_SWEEP}
    return lm_edit, base_sd, after_sd, dkeys, builders, edited_params, ldesc


def _score_task(task, rows, per_cfg_preds, n_boot):
    """Compute m(cfg), metric CI, R_cap(cfg) point + paired-bootstrap CI for one task. Per-item tasks
    use a PAIRED resample of example indices across cfg/FLOOR/CEILING (consistent draws); the corpus
    task (CIDEr) tokenizes once then paired-resamples pre-tokenized ids."""
    import numpy as np
    rng = np.random.default_rng(SEED)
    n = len(rows)
    corpus = _is_corpus(task)

    if corpus:
        tok = {c: _corpus_tokenized(rows, preds) for c, preds in per_cfg_preds.items()}
        m = {c: _cider_from_tokenized(*tok[c]) for c in per_cfg_preds}

        def _metric_on(c, ids):
            return _cider_from_tokenized(tok[c][0], tok[c][1], ids)
        nb = min(n_boot, N_BOOT_CORPUS)
    else:
        s = {c: _per_item_scores(task, rows, preds) for c, preds in per_cfg_preds.items()}
        m = {c: float(s[c].mean()) for c in per_cfg_preds}

        def _metric_on(c, ids):
            return float(s[c][ids].mean())
        nb = n_boot

    # metric CIs (resample the SAME draw indices across configs so they share noise)
    idx_draws = [rng.integers(0, n, n) for _ in range(nb)]
    out = {}
    for c in per_cfg_preds:
        vals = np.array([_metric_on(c, ids) for ids in idx_draws])
        out[c] = {"metric": m[c], "metric_ci": {"lo": float(np.percentile(vals, 2.5)),
                                                 "hi": float(np.percentile(vals, 97.5))}}
    # R_cap point + paired CI
    denom = m["CEILING"] - m["FLOOR"]
    for c in per_cfg_preds:
        rcap = (m[c] - m["FLOOR"]) / denom if denom > 0 else float("nan")
        rdraws = []
        for ids in idx_draws:
            mf, mk = _metric_on("FLOOR", ids), _metric_on("CEILING", ids)
            if mk > mf:
                rdraws.append((_metric_on(c, ids) - mf) / (mk - mf))
        out[c]["R_cap"] = rcap
        out[c]["R_cap_ci"] = ({"lo": float(np.percentile(rdraws, 2.5)),
                               "hi": float(np.percentile(rdraws, 97.5))} if rdraws else None)
    return {"corpus": corpus, "n": n, "denom_ceiling_minus_floor": denom, "configs": out}


def run_sweep(out, n, batch, n_boot, task_names, device):
    import torch
    from infra.pg import battery_tasks as B
    from infra.pg import diff_harvest as H

    pg_after, pg_before, proc, paths, _img, gate = H.build_pair("base-pt", device)
    print(f"[capreadout/sweep] swap_gate={json.dumps(gate, default=str)}", flush=True)
    lm_edit, base_sd, after_sd, dkeys, builders, edited_params, ldesc = _build_edit_family(
        pg_after, pg_before, paths, device)
    del pg_after                                            # sweep generation only edits pg_before
    torch.cuda.empty_cache()
    print(f"[capreadout/sweep] {len(builders)} configs: {[b[0] for b in builders]}", flush=True)

    results = {"experiment": "e4_capability_sweep", "status": "confirmatory", "seed": SEED,
               "config": {"k_sweep": list(K_SWEEP), "k_max": K_MAX, "n3_k": N3_K, "n": n,
                          "batch": batch, "n_boot": n_boot},
               "provenance": {"axis": "base-pt", "ceiling": "google/paligemma2-3b-pt-224",
                              "floor_decoder": "google/gemma-2-2b", "swap_gate": gate,
                              "rkl_compare_source": "s3://pg-lineage-769063704696/results/e4_basept_v2"},
               "edited_decoder_matrix_params": edited_params, "L_desc_params": ldesc, "tasks": {}}

    for tname in task_names:
        task = B.TASKS[tname]
        rows = _eval_rows(task, n, SEED)
        print(f"\n[capreadout/sweep] === task={tname} split={task.eval_split} n={len(rows)} "
              f"metric={task.metric_name} corpus={_is_corpus(task)} mnt={task.max_new_tokens} ===", flush=True)
        per_cfg_preds = {}
        for cname, edited, override in builders:
            _set_decoder(lm_edit, override, edited, device)
            preds = _generate(pg_before, proc, task, rows, device, batch)
            per_cfg_preds[cname] = preds
            qa = (float(B.TASKS[tname].score_fn(preds[0], rows[0])) if not _is_corpus(task) else None)
            print(f"[capreadout/sweep]   {tname}/{cname:10s} done  e.g. pred[0]={preds[0][:64]!r}"
                  + (f" s0={qa:.0f}" if qa is not None else ""), flush=True)
        scored = _score_task(task, rows, per_cfg_preds, n_boot)
        # echo the headline row per task
        rc = scored["configs"]
        print(f"[capreadout/sweep]   {tname} R_cap: "
              + "  ".join(f"{c}={rc[c]['R_cap']:.3f}" for c in
                          ["FLOOR", f"rank{min(K_SWEEP)}", f"rank{max(K_SWEEP)}", "CEILING"]
                          if c in rc), flush=True)
        results["tasks"][tname] = {"split": task.eval_split, "metric": task.metric_name,
                                   "metric_source": task.metric_source, **scored,
                                   "samples": {c: per_cfg_preds[c][:12] for c in per_cfg_preds}}

    (out / "e4_capability_sweep.json").write_text(json.dumps(results, indent=2))
    print("E4_CAP_SWEEP_OK", json.dumps(
        {t: {c: results["tasks"][t]["configs"][c]["R_cap"]
             for c in ("FLOOR", f"rank{min(K_SWEEP)}", "CEILING")}
         for t in results["tasks"]}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="preflight", choices=["preflight", "sweep"])
    ap.add_argument("--n", type=int, default=None, help="examples/anchor (preflight 100, sweep 2000)")
    ap.add_argument("--batch", type=int, default=GEN_BATCH)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--tasks", default="all", help="'all' or comma list of refcoco,textvqa,coco_captions")
    ap.add_argument("--smoke", action="store_true", help="tiny N + refcoco only — wiring + anchors")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    device = "cuda"
    assert torch.cuda.is_available(), "capability readout needs a GPU (generation)"
    print(f"[capreadout] mode={a.mode} gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} "
          f"smoke={a.smoke}", flush=True)

    if a.mode == "preflight":
        run_preflight(out, a.n or N_PREFLIGHT, a.batch, device)
    else:
        task_names = TASKS_ALL if a.tasks == "all" else tuple(t.strip() for t in a.tasks.split(","))
        if a.smoke:
            task_names, n = ("refcoco",), (a.n or 16)
        else:
            n = a.n or N_SWEEP
        run_sweep(out, n, a.batch, a.n_boot, task_names, device)


if __name__ == "__main__":
    main()
