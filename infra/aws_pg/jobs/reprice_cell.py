"""REPRICE cell runner — ONE {decoder, arm, rank, n, seed, eval-split} cell.

Mirrors `w17_cell` (same relearning cores) but **co-measures TWO axes in the SAME
blank-image forward**, paired per item:

  MCQ axis  : `mmlu_blankimg`  (choice-letter argmax — the W17/W14 readout)
  cloze axis: `mmlu_cloze`     (answer-content; banks per-candidate logprob+ntok)

  decoder=pg     : adapters on pg_after's language_model (the DELETED subject)
  decoder=random : adapters on the random-decoder FLOOR build
  arm=cap        : MMLU auxiliary_train ladder subset, single-letter CE
  arm=shuffled   : same subset, labels permuted (EDL≈0 floor)
  n=0            : adapters attached, ZERO training steps (data-free row)

Banks RAW per-item arrays for BOTH axes (and the cloze logprobs/ntoks); all S_x /
C_x / J / bootstraps are computed downstream in `reprice_unblind` from the banked
rows. Eval restricted to the frozen split via `item_filter` (parity, shared with
W16/W17). Fail-loud; cell is the re-run unit.

    python -m infra.aws_pg.jobs.reprice_cell --out <dir> --decoder pg --arm cap \
        --rank 512 --n 16000 --seed 1337 --eval-split confirmation \
        --lr 5e-6 --epochs 1 --batch 8 --warmup 0.03
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--decoder", choices=["pg", "random"], required=True)
    ap.add_argument("--arm", choices=["cap", "shuffled"], required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--eval-split", choices=["bucket", "confirmation"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--warmup", type=float, required=True)
    ap.add_argument("--decoder-seed", type=int, default=1337)
    ap.add_argument("--alpha", type=float, default=-1.0,
                    help="-1 → α=rank (the W17 frozen recipe α=r); pass explicitly to override")
    ap.add_argument("--n-per-subject", type=int, default=0, help="smoke cap (0 = full split)")
    args = ap.parse_args()
    if args.alpha < 0:
        args.alpha = float(args.rank)            # frozen recipe α=r (NOT 2r) — see reference memory

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
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "reprice_cell needs GPU."
    device = "cuda"
    n_per = args.n_per_subject or None
    cell_id = f"{args.decoder}_{args.arm}_r{args.rank}_n{args.n}_s{args.seed}_{args.eval_split}"
    print(f"[REPRICE-CELL] {cell_id} recipe lr={args.lr} ep={args.epochs} b={args.batch} α={args.alpha}")

    pred = W.is_grid_subset if args.eval_split == "bucket" else W.is_confirmation
    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)

    # ---- build the decoder side ----
    pg_after, pg_before, proc, paths, _it, _gate = H.build_before_after(device)
    if args.decoder == "pg":
        model = pg_after
        del pg_before
    else:
        model = H.build_random_after(pg_before, paths, device, seed=args.decoder_seed)
        del pg_after, pg_before
    torch.cuda.empty_cache()
    lm = model.get_submodule(paths["language_model"])

    # ---- adapters ----
    arep = Rl.attach_adapters(lm, rank=args.rank, seed=args.seed, alpha=args.alpha)
    model.to(device)
    print(f"[REPRICE-CELL] adapters: {arep['n_modules']} modules, {arep['trainable_params']:,} trainable")

    # ---- data + training ----
    train_rep = {"steps": 0, "final_loss": None, "skipped": "n=0 row"}
    if args.n > 0:
        ds = load_dataset(TE.MMLU_REPO, "all")
        pool = [dict(r) for r in ds["auxiliary_train"]]
        # dedup against the FULL eval population (all test items) — matches W17 and keeps
        # the nested-ladder property (subset(n1) ⊂ subset(n2)) identical across every cell,
        # independent of which split this cell scores.
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
    clz = TE.mmlu_cloze(model, tok, device, processor=proc, n_per_subject=n_per,
                        item_filter=pred)
    assert mcq["per_item_hash"] == clz["per_item_hash"], f"{cell_id}: MCQ/cloze pairing HALT"
    mcq_item = mcq["per_item"]

    row = {
        "cell_id": cell_id, "decoder": args.decoder, "arm": args.arm,
        "rank": args.rank, "n": args.n, "seed": args.seed, "eval_split": args.eval_split,
        "adapter": {k: arep[k] for k in ("n_modules", "trainable_params", "rank", "alpha")},
        "train": train_rep,
        "item_hash": clz["per_item_hash"],
        "correct_idx": clz["correct_idx"],
        "mcq": {"acc_pct": round(100 * sum(mcq_item) / len(mcq_item), 2),
                "n_items": len(mcq_item), "per_item": mcq_item},
        "cloze": {"acc_pct": clz["acc_pct"], "n_items": clz["n"],
                  "per_item": clz["per_item"],
                  "logprobs": clz["cloze_logprobs"], "ntoks": clz["cloze_ntoks"]},
        "recipe": {"lr": args.lr, "epochs": args.epochs, "batch": args.batch,
                   "warmup": args.warmup, "alpha": args.alpha},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    (out / f"cell_{cell_id}.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[REPRICE-CELL] {cell_id}: MCQ={row['mcq']['acc_pct']:.2f} cloze={clz['acc_pct']:.2f} "
          f"n={clz['n']} loss={train_rep.get('final_loss')}")


if __name__ == "__main__":
    main()
