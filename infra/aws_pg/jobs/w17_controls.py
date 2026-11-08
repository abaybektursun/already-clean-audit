"""W17 CONTROLS (prereg v2 §5 — recipe freeze + floors + δ_eq; ALL control-side).

Stages (GPU; HALT on any gate):
  1. BEFORE + FLOOR(1337) full-split `mmlu_choice_logits` → the W17 scale
     arrays; gates: exact-match ≥ 0.999 + |Δacc| ≤ 0.1 vs W16 controls,
     hashes identical (extraction stability on this branch).
  2. Recipe positive control (BEFORE side, cap-data, r=512 × n=16k × seeds
     {1337,1338,1339}): (a) fit — final-epoch mean letter-CE ≤ 0.5;
     (b) one-sided no-degradation — bucket MMLU ≥ unprobed-bucket − p95
     half-width. These three runs double as δ_eq 16k band sources.
  3. Stability control: retain-only on BEFORE (r=512, 16k-matched budget,
     seed 1337) — one-sided no-degradation on the bucket.
  4. Random-decoder cap-data δ sources: r=512 × {1k, 16k} × 3 seeds, bucket.
  5. δ_eq(1k) from the random@1k cell; δ_eq(16k) from random@16k + positive
     control (pooled). Frozen into controls.json with CONTROLS_OK.

    python -m infra.aws_pg.jobs.w17_controls --out <dir> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

W16_CONTROLS_S3 = "s3://pg-lineage-769063704696/results/w16_controls_a1/out/w16_controls.json"
SEEDS = (1337, 1338, 1339)
RECIPE = {"lr": 2e-5, "epochs": 3, "batch": 8, "warmup": 0.03}   # eq1 composite candidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--w16-controls-s3", default=W16_CONTROLS_S3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=RECIPE["lr"])
    ap.add_argument("--epochs", type=int, default=RECIPE["epochs"])
    ap.add_argument("--alpha-mult", type=float, default=2.0,
                    help="adapter alpha = alpha_mult × rank (recipe-search winner)")
    ap.add_argument("--fit-ce", type=float, default=0.8,
                    help="fit gate: last-epoch letter-CE ≤ this (amended form)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from infra.pg import relearn as R
    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as W
    from infra.pg import w17_verdict as V
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available()
    device = "cuda"
    n_per = 2 if args.smoke else None
    n16 = 64 if args.smoke else 16_000
    n1 = 32 if args.smoke else 1_000
    ep = 1 if args.smoke else args.epochs
    RECIPE["lr"] = args.lr
    RECIPE["epochs"] = args.epochs
    print(f"[W17-CTRL] torch={torch.__version__} smoke={args.smoke} "
          f"gpu={torch.cuda.get_device_name(0)}")

    w16_local = out / "w16_controls_cached.json"
    subprocess.run(["aws", "s3", "cp", args.w16_controls_s3, str(w16_local),
                    "--region", "us-east-1"], check=True)
    W16C = json.loads(w16_local.read_text())
    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)

    # ---- 1. scale arrays + extraction gates ----
    pg_after, pg_before, _p, paths, _i, build_gate = H.build_before_after(device)
    del pg_after
    torch.cuda.empty_cache()
    before_cl = TE.mmlu_choice_logits(pg_before, tok, device, n_per_subject=n_per)
    hashes = before_cl["per_item_hash"]
    pg_floor = H.build_random_after(pg_before, paths, device, seed=1337)
    floor_cl = TE.mmlu_choice_logits(pg_floor, tok, device, n_per_subject=n_per)
    del pg_floor
    torch.cuda.empty_cache()
    gates_ext = {}
    if not args.smoke:
        assert hashes == W16C["item_hashes"], "item order drift vs W16 — HALT"
        for name, cl in (("BEFORE", before_cl), ("FLOOR", floor_cl)):
            match = float(np.mean(np.array(cl["per_item"]) == np.array(W16C[name]["per_item"])))
            d = abs(cl["acc_pct"] - float(W16C[name]["acc_pct"]))
            gates_ext[name] = {"exact_match": match, "abs_acc_delta": d,
                               "ok": bool(match >= 0.999 and d <= 0.1)}
            print(f"[W17-CTRL][1] {name} match={match:.4f} |Δ|={d:.3f}")
        assert all(g["ok"] for g in gates_ext.values()), f"extraction HALT: {gates_ext}"
    b_pi, f_pi = before_cl["per_item"], floor_cl["per_item"]
    gb = W.is_grid_subset
    b_bucket = W.mask_by(b_pi, hashes, gb)
    f_bucket = W.mask_by(f_pi, hashes, gb)
    bucket_hashes = W.mask_by(hashes, hashes, gb)
    # one-sided no-degradation reference: BEFORE unprobed bucket acc − p95 halfwidth
    from infra.pg.w16_probes import _boot_draws
    bb = np.asarray(b_bucket, float)
    draws, _ = _boot_draws(lambda idx: float(bb[idx].mean()), bb.size, args.n_boot, 1337)
    halfwidth = float((np.percentile(draws, 97.5) - np.percentile(draws, 2.5)) / 2)
    floor_ref = float(bb.mean() - halfwidth)
    print(f"[W17-CTRL][1] BEFORE bucket={bb.mean():.4f} halfwidth={halfwidth:.4f} "
          f"no-degradation floor={floor_ref:.4f}")

    # ---- data pool (shared) ----
    ds = load_dataset(TE.MMLU_REPO, "all")
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ded = R.dedup_against_eval(pool, hashes)
    print(f"[W17-CTRL] dedup: {ded['n_dropped']}/{ded['n_in']} dropped")

    def cap_examples(n):
        return [R.format_qa(r["question"], r["choices"], r["answer"])
                for r in R.ladder_subset(ded["kept"], n)]

    def train_eval(model, lm, examples, seed):
        rep = R.attach_adapters(lm, rank=512, seed=seed, alpha=args.alpha_mult * 512)
        model.to(device)
        tr = R.train_adapter_qa(model, tok, device, examples, lr=RECIPE["lr"],
                                epochs=ep, batch_size=RECIPE["batch"],
                                warmup_ratio=RECIPE["warmup"], seed=seed,
                                micro_batch=2)   # math-equivalent accumulation (OOM envelope)
        ev = TE.mmlu_choice_logits(model, tok, device, n_per_subject=n_per, item_filter=gb)
        assert args.smoke or ev["per_item_hash"] == bucket_hashes
        return rep, tr, ev

    # ---- 2. positive control: BEFORE-side cap r512 × 16k × 3 seeds ----
    ex16 = cap_examples(n16)
    pos = {}
    pos_arrays = []
    for seed in (SEEDS if not args.smoke else SEEDS[:1]):
        del pg_before                       # fresh, unmutated build per training
        torch.cuda.empty_cache()
        _pa, pg_before, _p2, paths, _i2, _g2 = H.build_before_after(device)
        del _pa
        torch.cuda.empty_cache()
        lm = pg_before.get_submodule(paths["language_model"])
        _, tr, ev = train_eval(pg_before, lm, ex16, seed)
        fit_ok = bool(tr["mean_loss_last_epoch"] <= args.fit_ce)
        nodeg_ok = bool(ev["acc"] >= floor_ref) if not args.smoke else True
        pos[seed] = {"train": tr, "bucket_acc": ev["acc"], "acc_pct": ev["acc_pct"],
                     "fit_ok": fit_ok, "no_degradation_ok": nodeg_ok,
                     "per_item": ev["per_item"]}
        pos_arrays.append(ev["per_item"])
        print(f"[W17-CTRL][2] pos seed={seed}: loss={tr['mean_loss_last_epoch']:.4f} "
              f"bucket={ev['acc_pct']:.2f} fit={fit_ok} nodeg={nodeg_ok}")
    recipe_ok = all(p["fit_ok"] and p["no_degradation_ok"] for p in pos.values())
    assert args.smoke or recipe_ok, f"RECIPE FREEZE HALT: {pos}"

    # ---- 3. stability: retain-only on BEFORE (budget of the 16k cell) ----
    budget = sum(len(tok.encode(e["prompt"], add_special_tokens=False)) + 1
                 for e in ex16) * ep
    cap_steps = len(R.qa_train_step_batches(ex16, RECIPE["batch"])) * ep
    wt = load_dataset(TE.WIKITEXT_REPO, "wikitext-103-raw-v1", split="train")
    texts = [t for t in wt["text"][:600_000] if t.strip()]
    sl = R.retain_token_matched_slice(texts, tok, budget)
    del pg_before
    torch.cuda.empty_cache()
    _pa, pg_before, _p3, paths, _i3, _g3 = H.build_before_after(device)
    del _pa
    torch.cuda.empty_cache()
    lm = pg_before.get_submodule(paths["language_model"])
    R.attach_adapters(lm, rank=512, seed=1337, alpha=2.0 * 512)
    pg_before.to(device)
    from infra.aws_pg.jobs.w17_cell import retain_lm_train
    rt = retain_lm_train(pg_before, tok, device, sl["texts"], lr=RECIPE["lr"],
                         max_steps=cap_steps if not args.smoke else 3,
                         batch_size=max(2, RECIPE["batch"] // 2),
                         warmup_ratio=RECIPE["warmup"], seed=1337,
                         window=256)   # memory envelope (smoke OOM incident)
    rt_ev = TE.mmlu_choice_logits(pg_before, tok, device, n_per_subject=n_per, item_filter=gb)
    stab_ok = bool(rt_ev["acc"] >= floor_ref) if not args.smoke else True
    print(f"[W17-CTRL][3] retain stability: bucket={rt_ev['acc_pct']:.2f} ok={stab_ok}")
    assert args.smoke or stab_ok, "STABILITY HALT: retain training degraded BEFORE"
    del pg_before
    torch.cuda.empty_cache()

    # ---- 4. random-decoder δ sources: cap r512 × {1k,16k} × 3 seeds ----
    rnd = {"1k": [], "16k": []}
    rnd_meta = {}
    for n_name, n_val in (("1k", n1), ("16k", n16)):
        exn = cap_examples(n_val)
        for seed in (SEEDS if not args.smoke else SEEDS[:2]):
            _pa, pg_b2, _p4, paths, _i4, _g4 = H.build_before_after(device)
            del _pa
            torch.cuda.empty_cache()
            pg_rnd = H.build_random_after(pg_b2, paths, device, seed=1337)
            del pg_b2
            torch.cuda.empty_cache()
            lm = pg_rnd.get_submodule(paths["language_model"])
            _, tr, ev = train_eval(pg_rnd, lm, exn, seed)
            rnd[n_name].append(ev["per_item"])
            rnd_meta[f"{n_name}_s{seed}"] = {"acc_pct": ev["acc_pct"],
                                             "loss": tr["mean_loss_last_epoch"]}
            print(f"[W17-CTRL][4] random {n_name} seed={seed}: bucket={ev['acc_pct']:.2f}")
            del pg_rnd
            torch.cuda.empty_cache()

    # ---- 5. δ_eq per n* (pooled band sources, prereg §5) ----
    deq_1k = V.delta_eq([rnd["1k"]], b_bucket, f_bucket, n_boot=args.n_boot, seed=1337)
    src_16k = [rnd["16k"]] + ([pos_arrays] if len(pos_arrays) >= 2 else [])  # smoke runs 1 pos seed
    deq_16k = V.delta_eq(src_16k, b_bucket, f_bucket, n_boot=args.n_boot, seed=1337)
    print(f"[W17-CTRL][5] δ_eq(1k)={deq_1k['delta_eq']:.4f} δ_eq(16k)={deq_16k['delta_eq']:.4f}")

    controls_ok = bool(args.smoke or (recipe_ok and stab_ok
                                      and all(g["ok"] for g in gates_ext.values())))
    result = {
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "smoke": args.smoke, "build_gate": build_gate,
        "item_hashes": hashes, "correct_idx": before_cl["correct_idx"],
        "BEFORE": {"acc_pct": before_cl["acc_pct"], "per_item": b_pi},
        "FLOOR": {"acc_pct": floor_cl["acc_pct"], "per_item": f_pi, "decoder_seed": 1337},
        "gates_extraction": gates_ext,
        "no_degradation_floor_bucket": floor_ref, "bucket_halfwidth": halfwidth,
        "dedup": {"n_in": ded["n_in"], "n_dropped": ded["n_dropped"]},
        "recipe": {**RECIPE, "alpha_mult": args.alpha_mult,
                   "alpha": f"{args.alpha_mult:g}r", "rank_primary": 512,
                   "epochs_used": ep, "FROZEN": True},
        "positive_control": {str(k): {kk: vv for kk, vv in v.items() if kk != "per_item"}
                             for k, v in pos.items()},
        "stability_retain": {"bucket_acc_pct": rt_ev["acc_pct"], "ok": stab_ok,
                             "train": rt, "budget_tokens": budget},
        "random_delta_sources": rnd_meta,
        "delta_eq": {"1k": deq_1k, "16k": deq_16k},
        "n_boot": args.n_boot,
        "CONTROLS_OK": controls_ok,
    }
    (out / "w17_controls.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[W17-CTRL] CONTROLS_OK={controls_ok}")
    if not controls_ok:
        raise RuntimeError("W17 CONTROLS HALT")


if __name__ == "__main__":
    main()
