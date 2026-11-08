"""SAV-A cell runner — ONE {checkpoint, n, seed, eval-split} cell (prereg §2).

checkpoint ∈ {PG1, pi0, pi05, random}: decoder via `vlat_retention._extract_decoder`
(PG1 LM tower / pi0 remap / pi05 sealed-S3-or-remap) or the scale-matched random
decoder; run through the Gemma-1 swap-back shell (`load_swapback_model`, eager,
strict-gated). Adapters r=512 α=r on the shell decoder scope; ONE frozen recipe;
capability arm only (the SAV design has no retain/shuffled arms). Eval:
`mmlu_choice_logits` with the gemma-2b tokenizer; S vs the frozen SAV controls
BASE/FLOOR arrays. Cell = re-run unit; fail-loud.

    python -m infra.aws_pg.jobs.sav_cell --out <dir> --controls-s3 <sav_controls.json> \
        --checkpoint PG1 --n 16000 --seed 1337 --eval-split confirmation \
        --lr 5e-6 --epochs 1 --batch 8 --warmup 0.03 [--pi05-sealed-s3 s3://...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--checkpoint", choices=["PG1", "pi0", "pi05", "random"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--eval-split", choices=["bucket", "confirmation"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--warmup", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=-1.0, help="-1 → α=rank")
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--decoder-seed", type=int, default=1337, help="random-checkpoint draw")
    ap.add_argument("--pi05-sealed-s3", default="")
    ap.add_argument("--n-per-subject", type=int, default=0)
    args = ap.parse_args()
    if args.alpha < 0:
        args.alpha = float(args.rank)          # α = r (SAV frozen convention)
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
    from infra.pg.entities import ENTITIES
    from infra.pg.swap_back import boot_s_text, s_text
    from infra.aws_pg.jobs.vlat_retention import _bare_decoder, _extract_decoder, _random_decoder_sd

    assert torch.cuda.is_available()
    device = "cuda"
    n_per = args.n_per_subject or None
    cell_id = f"{args.checkpoint}_cap_r{args.rank}_n{args.n}_s{args.seed}_{args.eval_split}"
    print(f"[SAV-CELL] {cell_id} lr={args.lr} ep={args.epochs} α={args.alpha:g}")

    if args.controls_s3.startswith("s3://"):
        ctrl_local = out / "sav_controls.json"
        subprocess.run(["aws", "s3", "cp", args.controls_s3, str(ctrl_local),
                        "--region", "us-east-1"], check=True)
    else:
        ctrl_local = pathlib.Path(args.controls_s3)
    C = json.loads(ctrl_local.read_text())
    assert C["CONTROLS_OK"] is True and C["recipe"]["FROZEN"] is True
    fr = C["recipe"]
    assert (args.lr, args.epochs, args.batch, args.warmup) == \
           (fr["lr"], fr["epochs"], fr["batch"], fr["warmup"]), \
        f"cell args differ from FROZEN recipe: {(args.lr, args.epochs, args.batch, args.warmup)} vs {fr}"
    hashes = C["item_hashes"]
    b_pi, f_pi = C["BASE"]["per_item"], C["FLOOR"]["per_item"]
    pred = W.is_grid_subset if args.eval_split == "bucket" else W.is_confirmation
    b_m = W.mask_by(b_pi, hashes, pred)
    f_m = W.mask_by(f_pi, hashes, pred)
    split_hashes = W.mask_by(hashes, hashes, pred)

    ent = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(ent.repo_id, revision=ent.revision)

    # ---- build the shell for this checkpoint ----
    pi05_local = ""
    if args.checkpoint == "pi05" and args.pi05_sealed_s3:
        pi05_local = str(out / "pi05_decoder.safetensors")
        subprocess.run(["aws", "s3", "cp", args.pi05_sealed_s3, pi05_local,
                        "--region", "us-east-1"], check=True)
    if args.checkpoint == "random":
        base = AutoModelForCausalLM.from_pretrained(
            ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
            device_map="cpu", attn_implementation="eager")
        dec = _random_decoder_sd(_bare_decoder(base.state_dict()), args.decoder_seed)
        del base
        src_sha = f"random(seed={args.decoder_seed})"
    else:
        dec, src_sha = _extract_decoder(args.checkpoint, device, pi05_local)
        assert dec is not None, f"{args.checkpoint}: extractor returned no decoder"
        if args.checkpoint == "pi05":
            src_sha = f"{src_sha}|{'sealed:' + args.pi05_sealed_s3 if args.pi05_sealed_s3 else 'fresh-remap'}"
    model, swap_info = SB.load_swapback_model(dec, device)
    del dec
    torch.cuda.empty_cache()
    if pi05_local and pathlib.Path(pi05_local).exists():
        pathlib.Path(pi05_local).unlink()      # 7.4 GB per step accumulates → disk-full (s1339 incident)

    # ---- adapters + data + training ----
    arep = R.attach_adapters(model.model, rank=args.rank, seed=args.seed, alpha=args.alpha)
    model.to(device)
    train_rep = {"steps": 0, "skipped": "n=0 row"}
    if args.n > 0:
        ds = load_dataset(TE.MMLU_REPO, "all")
        pool = [dict(r) for r in ds["auxiliary_train"]]
        ded = R.dedup_against_eval(pool, hashes)
        sub = R.ladder_subset(ded["kept"], args.n)
        examples = [R.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
        train_rep = R.train_adapter_qa(model, tok, device, examples, lr=args.lr,
                                       epochs=args.epochs, batch_size=args.batch,
                                       warmup_ratio=args.warmup, seed=args.seed,
                                       micro_batch=2)
        train_rep["dedup"] = {"n_in": ded["n_in"], "n_dropped": ded["n_dropped"]}

    # ---- eval (+ descriptive ppl at the 16k cells, prereg §2) ----
    ppl = None
    if args.n == 16000 and not n_per:
        ppl = TE.clean_text_perplexity(model, tok, device, max_tokens=60_000)["ppl"]
    ev = TE.mmlu_choice_logits(model, tok, device, n_per_subject=n_per, item_filter=pred)
    assert ev["per_item_hash"] == split_hashes, f"{cell_id}: item order != controls — HALT"
    S = s_text(ev["per_item"], b_m, f_m)
    Sb = boot_s_text(ev["per_item"], b_m, f_m)
    margins = W.margin_per_item(ev["choice_logits"], ev["correct_idx"])
    spread = float(np.mean([np.std(r) for r in np.asarray(ev["choice_logits"], float)]))

    row = {"cell_id": cell_id, "checkpoint": args.checkpoint, "n": args.n,
           "seed": args.seed, "eval_split": args.eval_split,
           "decoder_source_sha": str(src_sha), "swap_gate": {k: v for k, v in swap_info.items()},
           "adapter": {k: arep[k] for k in ("n_modules", "trainable_params", "rank", "alpha")},
           "train": train_rep,
           "eval": {"acc_pct": ev["acc_pct"], "n_items": ev["n"], "S_recover": S,
                    "S_boot": {"mean": Sb[0], "lo95": Sb[1], "hi95": Sb[2]},
                    "rank_auc": W.rank_auc(ev["choice_logits"], ev["correct_idx"]),
                    "margin_mean": float(margins.mean()),
                    "brier_mean": float(W.brier_per_item(ev["choice_logits"], ev["correct_idx"]).mean()),
                    "letter_logit_spread": spread, "wikitext_ppl": ppl},
           "per_item": ev["per_item"], "choice_logits": ev["choice_logits"],
           "per_item_hash": ev["per_item_hash"], "correct_idx": ev["correct_idx"],
           "controls_s3": args.controls_s3, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0)}
    (out / f"cell_{cell_id}.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[SAV-CELL] {cell_id}: acc={ev['acc_pct']:.2f} S={S:.4f} [{Sb[1]:.4f},{Sb[2]:.4f}]")


if __name__ == "__main__":
    main()
