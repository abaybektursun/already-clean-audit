"""W18 cell runner — ONE {model, rank, n, seed, eval-split} cell on the engineered system.

model ∈ {rmu, base, floor} (zephyr_shell loaders; base only for the positive
control); adapters rank∈{8,64,512} α=r on `model.model` (Mistral decoder scope);
ONE frozen recipe; relearn pool = `redwoodresearch/wmdp-cyber-deduped` splits
0–2 (375 rows); eval = the RTT-clean HELD-OUT population (splits 3–4+dev,
verbatim-mapped; read once) or the labeled complement — FATAL-1 design. S vs the frozen W18 controls BASE/FLOOR arrays.

    python -m infra.aws_pg.jobs.w18_cell --out <dir> --controls-s3 <w18_controls.json> \
        --model rmu --rank 512 --n 500 --seed 1337 --eval-split confirmation \
        --lr <frozen> --epochs <frozen> --batch 8 --warmup 0.03
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess


def pool_rows():
    """The frozen relearn pool: wmdp-cyber-deduped splits 0–2 (375 rows).
    (Critic FATAL-1 fold: splits 3–4 + dev are the RTT-clean HELD-OUT eval.)"""
    from datasets import load_dataset
    ds = load_dataset("redwoodresearch/wmdp-cyber-deduped")
    rows = []
    for sp in ("split_0", "split_1", "split_2"):
        rows.extend(dict(r) for r in ds[sp])
    return rows


def heldout_rows():
    """The RTT-clean held-out eval source: splits 3–4 + dev (255 rows)."""
    from datasets import load_dataset
    ds = load_dataset("redwoodresearch/wmdp-cyber-deduped")
    rows = []
    for sp in ("split_3", "split_4", "dev"):
        rows.extend(dict(r) for r in ds[sp])
    return rows


def pool_hashes(rows):
    return {hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--model", choices=["rmu", "base", "floor"], required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--eval-split", choices=["heldout", "complement"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--warmup", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=-1.0)
    ap.add_argument("--floor-seed", type=int, default=1337)
    args = ap.parse_args()
    if args.alpha < 0:
        args.alpha = float(args.rank)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from infra.pg import relearn as R
    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as W
    from infra.pg import zephyr_shell as Z
    from infra.pg.swap_back import boot_s_text, s_text

    assert torch.cuda.is_available()
    device = "cuda"
    cell_id = f"{args.model}_cap_r{args.rank}_n{args.n}_s{args.seed}_{args.eval_split}"
    print(f"[W18-CELL] {cell_id} lr={args.lr} ep={args.epochs}")

    if args.controls_s3.startswith("s3://"):
        ctrl_local = out / "w18_controls.json"
        subprocess.run(["aws", "s3", "cp", args.controls_s3, str(ctrl_local),
                        "--region", "us-east-1"], check=True)
    else:
        ctrl_local = pathlib.Path(args.controls_s3)
    C = json.loads(ctrl_local.read_text())
    assert C["CONTROLS_OK"] is True and C["recipe"]["FROZEN"] is True
    fr = C["recipe"]
    assert (args.lr, args.epochs, args.batch, args.warmup) == \
           (fr["lr"], fr["epochs"], fr["batch"], fr["warmup"]), "recipe mismatch vs FROZEN"
    hashes = C["item_hashes"]                 # the DEDUPED eval population, frozen order
    b_pi, f_pi = C["BASE"]["per_item"], C["FLOOR"]["per_item"]
    pool_h = set(C["pool_hashes"])

    # eval-split semantics (FATAL-1 fold): "heldout" = the RTT-clean deduped
    # held-out population (controls' heldout_hashes; read ONCE — 0.5 is an
    # absolute threshold, no tuning split); "complement" = the leakage-upper-
    # bound descriptive population.
    heldout_h = set(C["heldout_hashes"])
    if args.eval_split == "heldout":
        pred_split = lambda h: h in heldout_h
    elif args.eval_split == "complement":
        pred_split = lambda h: h not in heldout_h and h not in pool_h
    else:
        raise SystemExit(f"unknown eval-split {args.eval_split}")

    def eval_pred(h):
        return pred_split(h)

    b_m = [v for v, h in zip(b_pi, hashes) if pred_split(h)]
    f_m = [v for v, h in zip(f_pi, hashes) if pred_split(h)]
    split_hashes = [h for h in hashes if pred_split(h)]

    tok = AutoTokenizer.from_pretrained(Z.ZEPHYR_BASE)
    R.letter_token_ids(tok)

    model, info = Z.load_zephyr(args.model, device, floor_seed=args.floor_seed)
    arep = R.attach_adapters(model.model, rank=args.rank, seed=args.seed, alpha=args.alpha)
    model.to(device)

    train_rep = {"steps": 0, "skipped": "n=0 row"}
    if args.n > 0:
        rows = pool_rows()
        assert len(rows) == 375, f"pool size {len(rows)} != 375 — provenance drift"
        sub = R.ladder_subset(rows, args.n, salt="w18-ladder")
        examples = [R.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
        train_rep = R.train_adapter_qa(model, tok, device, examples, lr=args.lr,
                                       epochs=args.epochs, batch_size=args.batch,
                                       warmup_ratio=args.warmup, seed=args.seed,
                                       micro_batch=2)

    ev = TE.wmdp_choice_logits(model, tok, device, config="wmdp-cyber", item_filter=eval_pred)
    assert ev["per_item_hash"] == split_hashes, f"{cell_id}: eval items != controls — HALT"
    S = s_text(ev["per_item"], b_m, f_m)
    Sb = boot_s_text(ev["per_item"], b_m, f_m)
    row = {"cell_id": cell_id, "model": args.model, "rank": args.rank, "n": args.n,
           "seed": args.seed, "eval_split": args.eval_split, "shell_info": info,
           "adapter": {k: arep[k] for k in ("n_modules", "trainable_params", "rank", "alpha")},
           "train": train_rep,
           "eval": {"acc_pct": ev["acc_pct"], "n_items": ev["n"], "S_recover": S,
                    "S_boot": {"mean": Sb[0], "lo95": Sb[1], "hi95": Sb[2]},
                    "rank_auc": W.rank_auc(ev["choice_logits"], ev["correct_idx"])},
           "per_item": ev["per_item"], "choice_logits": ev["choice_logits"],
           "per_item_hash": ev["per_item_hash"], "correct_idx": ev["correct_idx"],
           "controls_s3": args.controls_s3, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0)}
    (out / f"cell_{cell_id}.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[W18-CELL] {cell_id}: acc={ev['acc_pct']:.2f} S={S:.4f} [{Sb[1]:.4f},{Sb[2]:.4f}]")


if __name__ == "__main__":
    main()
