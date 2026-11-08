"""LATMECH cell runner — ONE {condition, arm, rank, n, seed, eval-split} cell.

Extends `reprice_cell` (same relearning + co-measured MCQ/cloze spine) with the LATMECH
WEIGHT CONDITIONS the MDE contrasts (prereg §3.1/§3.2). Each cell: build the decoder under
one condition, attach adapters, relearn `n` MMLU examples, co-measure MCQ + cloze (blank-image,
same forward, paired per item), bank RAW per-item arrays. All S_x / RF / Δ / bootstraps are
computed downstream in `latmech_unblind` from the banked rows.

  condition=deleted        : pg_after, unmodified (content preserved — REPRICE). The reference.
  condition=content        : project the cloze-content carrier OUT of every decoder matrix
  condition=emrand         : project an ENERGY-MATCHED RANDOM subspace out (the MDE denominator)
  condition=shrink_perturb : W' = λ·W + σ·N(0, std(W))  (the basin arm; --sp-lam --sp-sigma)
  condition=random_floor   : the random-decoder FLOOR (no content, no savings — positive control)
  condition=interp         : (1−t)·deleted + t·floor  (Arm-2 spectrum; --spectrum-frac t)
  condition=lobotomy       : reinit one mid LM layer (dynamic-range positive control; --lobotomy-layer)
  arm=cap / shuffled       : MMLU auxiliary_train ladder subset, real / permuted labels
  n=0                      : adapters attached, ZERO training steps (data-free row)

content/emrand require --carrier (the carrier.pt cached by latmech_preflight). Fail-loud; the
cell is the re-run unit.

    python -m infra.aws_pg.jobs.latmech_cell --out <dir> --condition content --arm cap \
        --rank 512 --n 16000 --seed 1337 --eval-split confirmation --carrier carrier.pt \
        --lr 5e-6 --epochs 1 --batch 8 --warmup 0.03
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", required=True,
                    choices=["deleted", "content", "emrand", "shrink_perturb",
                             "random_floor", "interp", "lobotomy"])
    ap.add_argument("--arm", choices=["cap", "shuffled"], default="cap")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--eval-split", choices=["bucket", "confirmation"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--warmup", type=float, required=True)
    ap.add_argument("--carrier", default="", help="path to carrier.pt (content/emrand conditions)")
    ap.add_argument("--sp-lam", type=float, default=1.0, help="shrink_perturb λ")
    ap.add_argument("--sp-sigma", type=float, default=0.0, help="shrink_perturb σ")
    ap.add_argument("--spectrum-frac", type=float, default=0.0, help="interp t (deleted↔floor)")
    ap.add_argument("--lobotomy-layer", type=int, default=8)
    ap.add_argument("--decoder-seed", type=int, default=1337)
    ap.add_argument("--alpha", type=float, default=-1.0,
                    help="-1 → α=rank (frozen recipe α=r); pass explicitly to override")
    ap.add_argument("--n-per-subject", type=int, default=0, help="smoke cap (0 = full split)")
    args = ap.parse_args()
    if args.alpha < 0:
        args.alpha = float(args.rank)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from infra.pg import relearn as Rl
    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as W
    from infra.pg import latmech as Lm
    from infra.pg import w14_localize as W14
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs import _w15_shared as S15

    assert torch.cuda.is_available(), "latmech_cell needs GPU."
    device = "cuda"
    n_per = args.n_per_subject or None
    cell_id = (f"{args.condition}_{args.arm}_r{args.rank}_n{args.n}_s{args.seed}_"
               f"{args.eval_split}")
    print(f"[LATMECH-CELL] {cell_id} recipe lr={args.lr} ep={args.epochs} b={args.batch} α={args.alpha}")

    pred = W.is_grid_subset if args.eval_split == "bucket" else W.is_confirmation
    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    dkeys = W14.decoder_keys()

    # ---- build the decoder side under the requested condition ----
    pg_after, pg_before, proc, paths, _it, _gate = H.build_before_after(device)
    cond_info = {"condition": args.condition}
    if args.condition == "random_floor":
        model = H.build_random_after(pg_before, paths, device, seed=args.decoder_seed)
        del pg_after, pg_before
    else:
        model = pg_after
        lm = model.get_submodule(paths["language_model"])
        if args.condition in ("content", "emrand"):
            assert args.carrier, f"{args.condition} requires --carrier carrier.pt (local path or s3://)"
            carrier_path = args.carrier
            if carrier_path.startswith("s3://"):
                import subprocess
                carrier_path = str(out / "carrier.pt")
                subprocess.run(["aws", "s3", "cp", args.carrier, carrier_path,
                                "--region", "us-east-1"], check=True)
            factors = torch.load(carrier_path, map_location="cpu")
            which = "carrier" if args.condition == "content" else "emrand"
            base_sd = {k: v.detach() for k, v in lm.state_dict().items()}
            n_ov = S15.apply_carrier(lm, base_sd, dkeys, factors, which, device)
            cond_info["n_overridden"] = n_ov
            cond_info["removed_E"] = sum(factors[k]["removed_E"] for k in dkeys)
            cond_info["emrand_E"] = sum(factors[k]["emrand_E"] for k in dkeys)
            del pg_before
        elif args.condition == "shrink_perturb":
            # H-basin: per-matrix magnitude-MATCHED to the content ablation (§5). Load the carrier for
            # each matrix's removed energy; perturb so ‖W'−W‖_F == sqrt(removed_E[k]) (λ from --sp-lam,
            # default 1.0 → matched noise). Makes the basin move magnitude-comparable to content/emrand
            # (not a free-energy artifact, §8-T3).
            assert args.carrier, "shrink_perturb requires --carrier (per-matrix content magnitude match)"
            cpath = args.carrier
            if cpath.startswith("s3://"):
                import subprocess
                cpath = str(out / "carrier.pt")
                subprocess.run(["aws", "s3", "cp", args.carrier, cpath, "--region", "us-east-1"],
                               check=True)
            cfac = torch.load(cpath, map_location="cpu")
            base_sd = {k: v.detach() for k, v in lm.state_dict().items()}
            g = torch.Generator(device=device).manual_seed(args.seed)
            ov, tot2 = {}, 0.0
            for k in dkeys:
                Wm = lm.get_parameter(k).detach().float().to(device)
                target_fro = float(cfac[k]["removed_E"]) ** 0.5
                noise = torch.randn(Wm.shape, generator=g, device=device, dtype=torch.float32) * Wm.std()
                cur = float(noise.norm())
                if cur > 0:
                    noise = noise * (target_fro / cur)
                ov[k] = (args.sp_lam * Wm + noise).to(torch.bfloat16)
                tot2 += target_fro ** 2
            from infra.aws_pg.jobs.e4_compress_curve import _set_decoder
            _set_decoder(lm, base_sd, ov, device)
            cond_info.update({"sp_lam": args.sp_lam, "matched_total_fro": tot2 ** 0.5})
            del pg_before
        elif args.condition == "interp":
            floor = H.build_random_after(pg_before, paths, device, seed=args.decoder_seed)
            floor_lm = floor.get_submodule(paths["language_model"])
            floor_sd = {k: floor_lm.get_parameter(k).detach().float().cpu() for k in dkeys}
            del floor, pg_before
            torch.cuda.empty_cache()
            base_sd = {k: v.detach() for k, v in lm.state_dict().items()}
            ov = {}
            for k in dkeys:
                Wd = lm.get_parameter(k).detach().float().to(device)
                ov[k] = Lm.interpolate_weights(Wd, floor_sd[k].to(device),
                                               args.spectrum_frac).to(device).to(torch.bfloat16)
            from infra.aws_pg.jobs.e4_compress_curve import _set_decoder
            _set_decoder(lm, base_sd, ov, device)
            cond_info["spectrum_frac"] = args.spectrum_frac
        elif args.condition == "lobotomy":
            W.reinit_lm_layer_(lm, args.lobotomy_layer, seed=args.decoder_seed)
            cond_info["lobotomy_layer"] = args.lobotomy_layer
            del pg_before
        else:  # deleted
            del pg_before
    torch.cuda.empty_cache()
    lm = model.get_submodule(paths["language_model"])

    # ---- adapters ----
    arep = Rl.attach_adapters(lm, rank=args.rank, seed=args.seed, alpha=args.alpha)
    model.to(device)
    print(f"[LATMECH-CELL] adapters: {arep['n_modules']} modules, {arep['trainable_params']:,} trainable")

    # ---- data + relearning ----
    train_rep = {"steps": 0, "final_loss": None, "skipped": "n=0 row"}
    if args.n > 0:
        ds = load_dataset(TE.MMLU_REPO, "all")
        pool = [dict(r) for r in ds["auxiliary_train"]]
        ev_hashes = [Rl.qa_item_hash(r["question"]) for r in ds["test"]]
        ded = Rl.dedup_against_eval(pool, ev_hashes)
        sub = Rl.ladder_subset(ded["kept"], args.n)
        if args.arm == "shuffled":
            sub = Rl.shuffle_labels(sub, seed=args.seed)
        examples = [Rl.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
        train_rep = Rl.train_adapter_qa(model, tok, device, examples, lr=args.lr,
                                        epochs=args.epochs, batch_size=args.batch,
                                        warmup_ratio=args.warmup, seed=args.seed, micro_batch=2)
        train_rep["dedup"] = {"n_in": ded["n_in"], "n_dropped": ded["n_dropped"]}

    # ---- co-measure: MCQ + cloze, blank-image, SAME split, paired per item ----
    mcq = TE.mmlu_blankimg(model, proc, device, n_per_subject=n_per, item_filter=pred)
    clz = TE.mmlu_cloze(model, tok, device, processor=proc, n_per_subject=n_per, item_filter=pred)
    Lm.position_align_tripwire(mcq["per_item_hash"], clz["per_item_hash"])
    mcq_item = mcq["per_item"]

    row = {
        "cell_id": cell_id, "condition_info": cond_info, "arm": args.arm,
        "rank": args.rank, "n": args.n, "seed": args.seed, "eval_split": args.eval_split,
        "adapter": {k: arep[k] for k in ("n_modules", "trainable_params", "rank", "alpha")},
        "train": train_rep,
        "item_hash": clz["per_item_hash"],
        "correct_idx": clz["correct_idx"],
        "mcq": {"acc_pct": round(100 * sum(mcq_item) / len(mcq_item), 2),
                "n_items": len(mcq_item), "per_item": mcq_item},
        "cloze": {"acc_pct": clz["acc_pct"], "n_items": clz["n"], "per_item": clz["per_item"],
                  "logprobs": clz["cloze_logprobs"], "ntoks": clz["cloze_ntoks"]},
        "recipe": {"lr": args.lr, "epochs": args.epochs, "batch": args.batch,
                   "warmup": args.warmup, "alpha": args.alpha},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    (out / f"cell_{cell_id}.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[LATMECH-CELL] {cell_id}: MCQ={row['mcq']['acc_pct']:.2f} cloze={clz['acc_pct']:.2f} "
          f"n={clz['n']} loss={train_rep.get('final_loss')}")


if __name__ == "__main__":
    main()
