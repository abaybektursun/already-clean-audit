"""SAV-A CONTROLS (prereg §5 — scale arrays, recipe freeze on BASE-shell, δ bands).

  1. BASE-shell (raw gemma-2b) + FLOOR-shell (random decoder, seed 1337) full-split
     `mmlu_choice_logits` → frozen scale arrays; denom gates on bucket + confirmation;
     FLOOR decoder seeds 1338/1339 bucket context.
  2. Recipe freeze (positive control, BASE-shell, cap 16k × 3 seeds): W17-winner
     CANDIDATE (lr 5e-6, 1 ep, α=r) — gates: fit (last-epoch CE ≤ 0.8) AND one-sided
     no-degradation (bucket ≥ unprobed − p95 halfwidth). HALT + pre-stated search
     ladder if it fails (W17 process, SAV's own data).
  3. Random-checkpoint δ sources: cap r=512 × {1k, 16k} × 3 adapter seeds, bucket.
  4. δ_level = δ_pair via `w17_verdict.delta_eq` (max-statistic, documented) over
     {random@1k×3, random@16k×3, positive×3} per the prereg unit note.

    python -m infra.aws_pg.jobs.sav_controls --out <dir> [--smoke] [--lr ... --epochs ... --alpha-mult ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib

SEEDS = (1337, 1338, 1339)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--alpha-mult", type=float, default=1.0)
    ap.add_argument("--fit-ce", type=float, default=0.8)
    ap.add_argument("--n-boot", type=int, default=2000)
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from infra.pg import relearn as R
    from infra.pg import text_eval as TE
    from infra.pg import vlat_swapback as SB
    from infra.pg import w16_probes as W
    from infra.pg import w17_verdict as V
    from infra.pg.entities import ENTITIES
    from infra.pg.w16_probes import _boot_draws
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder, _random_decoder_sd

    assert torch.cuda.is_available()
    device = "cuda"
    n_per = 2 if args.smoke else None
    n16 = 64 if args.smoke else 16_000
    n1 = 32 if args.smoke else 1_000
    print(f"[SAV-CTRL] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} smoke={args.smoke}")

    ent = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(ent.repo_id, revision=ent.revision)
    R.letter_token_ids(tok)                       # gemma-1 single-token letters, fail-loud

    def fresh_base(dev="cpu"):
        return AutoModelForCausalLM.from_pretrained(
            ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
            device_map=dev, attn_implementation="eager").eval()

    # ---- 1. scale arrays ----
    base = fresh_base(device)
    base_cl = TE.mmlu_choice_logits(base, tok, device, n_per_subject=n_per)
    hashes = base_cl["per_item_hash"]
    base_sd_bare = _bare_decoder(base.state_dict())
    del base
    torch.cuda.empty_cache()
    floor_ctx = {}
    floor_cl = None
    for fs in (SEEDS if not args.smoke else SEEDS[:1]):
        dec = _random_decoder_sd(base_sd_bare, fs)
        m, _info = SB.load_swapback_model(dec, device)
        if fs == 1337:
            floor_cl = TE.mmlu_choice_logits(m, tok, device, n_per_subject=n_per)
            print(f"[SAV-CTRL][1] FLOOR(1337) acc={floor_cl['acc_pct']:.2f}")
        else:
            ev = TE.mmlu_choice_logits(m, tok, device, n_per_subject=n_per, item_filter=W.is_grid_subset)
            floor_ctx[fs] = ev["acc_pct"]
        del m
        torch.cuda.empty_cache()
    assert floor_cl["per_item_hash"] == hashes and floor_cl["correct_idx"] == base_cl["correct_idx"]
    b_pi, f_pi = base_cl["per_item"], floor_cl["per_item"]
    gb = W.is_grid_subset
    b_bucket = W.mask_by(b_pi, hashes, gb)
    f_bucket = W.mask_by(f_pi, hashes, gb)
    scale = {}
    for sname, p in (("bucket", gb), ("confirmation", W.is_confirmation)):
        bm, fm = W.mask_by(b_pi, hashes, p), W.mask_by(f_pi, hashes, p)
        den = float(np.mean(bm) - np.mean(fm))
        scale[sname] = {"n": len(bm), "denom": den, "floor_pct": 100 * float(np.mean(fm)),
                        "ok": bool(den > 0.05)}
        print(f"[SAV-CTRL][1] scale {sname}: denom={den:.4f}")
    assert args.smoke or all(s["ok"] for s in scale.values())
    bb = np.asarray(b_bucket, float)
    draws, _ = _boot_draws(lambda idx: float(bb[idx].mean()), bb.size, args.n_boot, 1337)
    halfwidth = float((np.percentile(draws, 97.5) - np.percentile(draws, 2.5)) / 2)
    floor_ref = float(bb.mean() - halfwidth)
    print(f"[SAV-CTRL][1] BASE bucket={bb.mean():.4f} no-degradation floor={floor_ref:.4f}")

    # ---- shared data ----
    ds = load_dataset(TE.MMLU_REPO, "all")
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ded = R.dedup_against_eval(pool, hashes)

    def cap_examples(n):
        return [R.format_qa(r["question"], r["choices"], r["answer"])
                for r in R.ladder_subset(ded["kept"], n)]

    def train_eval_on(model, examples, seed):
        rep = R.attach_adapters(model.model, rank=512, seed=seed, alpha=args.alpha_mult * 512)
        model.to(device)
        tr = R.train_adapter_qa(model, tok, device, examples, lr=args.lr,
                                epochs=args.epochs, batch_size=args.batch,
                                warmup_ratio=args.warmup, seed=seed, micro_batch=2)
        ev = TE.mmlu_choice_logits(model, tok, device, n_per_subject=n_per, item_filter=gb)
        return rep, tr, ev

    # ---- 2. recipe positive control on BASE-shell ----
    ex16 = cap_examples(n16)
    pos, pos_arrays = {}, []
    for seed in (SEEDS if not args.smoke else SEEDS[:1]):
        m = fresh_base(device)
        _, tr, ev = train_eval_on(m, ex16, seed)
        fit_ok = bool(tr["mean_loss_last_epoch"] <= args.fit_ce)
        nodeg_ok = bool(ev["acc"] >= floor_ref) if not args.smoke else True
        pos[seed] = {"train": {k: tr[k] for k in ("steps", "final_loss", "mean_loss_first_epoch",
                                                  "mean_loss_last_epoch", "tokens_seen")},
                     "bucket_acc_pct": ev["acc_pct"], "fit_ok": fit_ok,
                     "no_degradation_ok": nodeg_ok}
        pos_arrays.append(ev["per_item"])
        print(f"[SAV-CTRL][2] pos seed={seed}: CE={tr['mean_loss_last_epoch']:.3f} "
              f"bucket={ev['acc_pct']:.2f} fit={fit_ok} nodeg={nodeg_ok}")
        del m
        torch.cuda.empty_cache()
    recipe_ok = all(p["fit_ok"] and p["no_degradation_ok"] for p in pos.values())
    assert args.smoke or recipe_ok, f"SAV RECIPE FREEZE HALT (run the W17 search ladder): {pos}"

    # ---- 3. random-checkpoint δ sources ----
    rnd = {"1k": [], "16k": []}
    rnd_meta = {}
    for n_name, n_val in (("1k", n1), ("16k", n16)):
        exn = cap_examples(n_val)
        for seed in (SEEDS if not args.smoke else SEEDS[:2]):
            dec = _random_decoder_sd(base_sd_bare, 1337)
            m, _i = SB.load_swapback_model(dec, device)
            _, tr, ev = train_eval_on(m, exn, seed)
            rnd[n_name].append(ev["per_item"])
            rnd_meta[f"{n_name}_s{seed}"] = {"acc_pct": ev["acc_pct"],
                                             "loss": tr["mean_loss_last_epoch"]}
            print(f"[SAV-CTRL][3] random {n_name} s{seed}: bucket={ev['acc_pct']:.2f}")
            del m
            torch.cuda.empty_cache()

    # ---- 4. δ band (W17 delta_eq construction, documented) ----
    src = [rnd["1k"], rnd["16k"]] + ([pos_arrays] if len(pos_arrays) >= 2 else [])
    deq = V.delta_eq(src, b_bucket, f_bucket, n_boot=args.n_boot, seed=1337)
    print(f"[SAV-CTRL][4] δ={deq['delta_eq']:.4f}")

    controls_ok = bool(args.smoke or (recipe_ok and all(s["ok"] for s in scale.values())))
    result = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "smoke": args.smoke, "item_hashes": hashes,
              "correct_idx": base_cl["correct_idx"],
              "BASE": {"acc_pct": base_cl["acc_pct"], "per_item": b_pi},
              "FLOOR": {"acc_pct": floor_cl["acc_pct"], "per_item": f_pi,
                        "decoder_seed": 1337, "ctx_bucket_acc": floor_ctx},
              "scale_validity": scale,
              "no_degradation_floor_bucket": floor_ref, "bucket_halfwidth": halfwidth,
              "dedup": {"n_in": ded["n_in"], "n_dropped": ded["n_dropped"]},
              "recipe": {"lr": args.lr, "epochs": args.epochs, "batch": args.batch,
                         "warmup": args.warmup, "alpha_mult": args.alpha_mult,
                         "fit_ce": args.fit_ce, "rank_primary": 512, "FROZEN": True},
              "positive_control": {str(k): v for k, v in pos.items()},
              "random_delta_sources": rnd_meta,
              "delta": {"level_and_pair": deq["delta_eq"],
                        "construction": "w17_verdict.delta_eq pooled-max p95 (documented)",
                        "n_draws_kept": deq["n_draws_kept"]},
              "CONTROLS_OK": controls_ok}
    (out / "sav_controls.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[SAV-CTRL] CONTROLS_OK={controls_ok}")
    if not controls_ok:
        raise RuntimeError("SAV CONTROLS HALT")


if __name__ == "__main__":
    main()
