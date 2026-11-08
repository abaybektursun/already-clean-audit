"""W17 cell runner — ONE {decoder, arm, rank, n, seed, eval-split} cell (prereg §2 unit).

Thin wrapper over the pure cores (`relearn`, `text_eval`, `w16_probes`):

  decoder=pg     : adapters on the released PG2's language_model (the deleted system)
  decoder=random : adapters on the random-decoder FLOOR build (no-prior ladder)
  arm=cap        : MMLU auxiliary_train ladder subset, single-letter CE (the curve)
  arm=shuffled   : same subset, labels permuted (EDL≈0 floor)
  arm=retain     : wikitext-103 slices, plain next-token LM loss, token/step-matched
                   to the SAME-n capability cell (budget computed analytically from the
                   ladder subset — deterministic, no cap-cell artifact needed)
  n=0            : adapters attached, ZERO training steps (the data-free row)

Eval: `mmlu_choice_logits` with the split's item filter; S_recover vs the FROZEN W17
controls' BEFORE/FLOOR arrays (masked per §2 convention); rank-AUC + margin/Brier banked.
Writes one machine-readable cell row. Fail-loud; cell is the re-run unit.

    python -m infra.aws_pg.jobs.w17_cell --out <dir> --controls-s3 <w17_controls.json> \
        --decoder pg --arm cap --rank 512 --n 1000 --seed 1337 --eval-split bucket \
        --lr 2e-5 --epochs 3 --batch 8 --warmup 0.03
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def retain_lm_train(model, tokenizer, device, texts, *, lr, max_steps, batch_size,
                    warmup_ratio, seed, window=512):
    """Retain-only trainer: standard next-token CE over fixed-length windows of the
    token-matched slice; same optimizer/schedule family as train_adapter_qa; step
    count matched to the capability cell. Fail-loud on non-finite loss."""
    import math

    import torch

    ids = []
    for t in texts:
        ids.extend(tokenizer.encode(t, add_special_tokens=False))
    chunks = [ids[i:i + window + 1] for i in range(0, len(ids) - 1, window)]
    chunks = [c for c in chunks if len(c) >= 2]
    assert chunks, "retain slice produced no windows"

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable params — attach adapters first"
    opt = torch.optim.AdamW(trainable, lr=lr)

    def f(s):
        nw = int(warmup_ratio * max_steps)
        if s < nw:
            return s / max(1, nw)
        p = (s - nw) / max(1, max_steps - nw)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, f)
    g = torch.Generator().manual_seed(int(seed))

    model.train()
    losses, step, tokens = [], 0, 0
    while step < max_steps:
        order = torch.randperm(len(chunks), generator=g).tolist()
        for s0 in range(0, len(order), batch_size):
            batch = [chunks[i] for i in order[s0:s0 + batch_size]]
            n = max(len(c) for c in batch)
            inp = torch.zeros(len(batch), n, dtype=torch.long)
            am = torch.zeros(len(batch), n, dtype=torch.long)
            for i, c in enumerate(batch):
                inp[i, :len(c)] = torch.tensor(c)
                am[i, :len(c)] = 1
            inp, am = inp.to(device), am.to(device)
            out = model(input_ids=inp[:, :-1], attention_mask=am[:, :-1])
            logits = out.logits.float()
            tgt = inp[:, 1:]
            mask = am[:, 1:].bool()
            loss = torch.nn.functional.cross_entropy(
                logits[mask], tgt[mask])
            assert torch.isfinite(loss), f"retain: non-finite loss at step {step} — HALT"
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
            tokens += int(mask.sum().item())
            step += 1
            if step >= max_steps:
                break
    model.eval()
    return {"steps": step, "final_loss": losses[-1], "tokens_seen": tokens,
            "recipe": {"lr": lr, "max_steps": max_steps, "batch_size": batch_size,
                       "warmup_ratio": warmup_ratio, "seed": int(seed), "window": window}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True, help="frozen w17_controls.json (s3:// or local)")
    ap.add_argument("--decoder", choices=["pg", "random", "before"], required=True,
                    help="pg=released (deleted) decoder; random=scratch FLOOR build; "
                         "before=intact pre-deletion decoder (§6.1 BEFORE-side positive control)")
    ap.add_argument("--arm", choices=["cap", "retain", "shuffled"], required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--eval-split", choices=["bucket", "confirmation"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--warmup", type=float, required=True)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="-1 → epochs-driven (default); >0 → train to exactly this many "
                         "steps regardless of epochs (Contrast B matched-train-CE arm)")
    ap.add_argument("--decoder-seed", type=int, default=-1,
                    help="-1 → controls' FLOOR decoder_seed (1337); 1338/1339 = the "
                         "DISCOVERY decoder-draw context cells (prereg §3)")
    ap.add_argument("--n-per-subject", type=int, default=0,
                    help="smoke/wiring cap (0 = full; decision runs use 0)")
    ap.add_argument("--alpha", type=float, default=-1.0,
                    help="-1 → α=2·rank (composite scale 2.0 at every rank — the eq1 "
                         "operating point; decouples the rank axis from effective LR)")
    args = ap.parse_args()
    if args.alpha < 0:
        args.alpha = 2.0 * args.rank

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
    from infra.pg.entities import ENTITIES
    from infra.pg.swap_back import boot_s_text, s_text

    assert torch.cuda.is_available(), "w17_cell needs GPU."
    device = "cuda"
    n_per = args.n_per_subject or None
    dtag = f"_d{args.decoder_seed}" if args.decoder_seed > 0 else ""
    cell_id = (f"{args.decoder}{dtag}_{args.arm}_r{args.rank}_n{args.n}_s{args.seed}_{args.eval_split}")
    print(f"[W17-CELL] {cell_id} recipe lr={args.lr} ep={args.epochs} b={args.batch}")

    # ---- frozen controls (scale arrays + hashes) ----
    if args.controls_s3.startswith("s3://"):
        ctrl_local = out / "w17_controls.json"
        subprocess.run(["aws", "s3", "cp", args.controls_s3, str(ctrl_local),
                        "--region", "us-east-1"], check=True)
    else:
        ctrl_local = pathlib.Path(args.controls_s3)
    C = json.loads(ctrl_local.read_text())
    assert C["CONTROLS_OK"] is True, "W17 controls not OK — do not run cells."
    hashes = C["item_hashes"]
    b_pi, f_pi = C["BEFORE"]["per_item"], C["FLOOR"]["per_item"]

    pred = W.is_grid_subset if args.eval_split == "bucket" else W.is_confirmation
    b_m = W.mask_by(b_pi, hashes, pred)
    f_m = W.mask_by(f_pi, hashes, pred)
    split_hashes = W.mask_by(hashes, hashes, pred)

    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)

    # ---- build the decoder side ----
    pg_after, pg_before, _proc, paths, _it, _gate = H.build_before_after(device)
    if args.decoder == "pg":
        model = pg_after
        del pg_before
    elif args.decoder == "before":
        model = pg_before          # intact pre-deletion decoder (§6.1 BEFORE-side positive control)
        del pg_after
    else:
        dseed = args.decoder_seed if args.decoder_seed > 0 else int(C["FLOOR"]["decoder_seed"])
        model = H.build_random_after(pg_before, paths, device, seed=dseed)
        del pg_after, pg_before
    torch.cuda.empty_cache()
    lm = model.get_submodule(paths["language_model"])

    # ---- adapters ----
    arep = R.attach_adapters(lm, rank=args.rank, seed=args.seed, alpha=args.alpha)
    model.to(device)
    print(f"[W17-CELL] adapters: {arep['n_modules']} modules, "
          f"{arep['trainable_params']:,} trainable")

    # ---- data + training ----
    train_rep = {"steps": 0, "final_loss": None, "tokens_seen": 0, "skipped": "n=0 row"}
    if args.n > 0:
        ds = load_dataset(TE.MMLU_REPO, "all")
        pool = [dict(r) for r in ds["auxiliary_train"]]
        ded = R.dedup_against_eval(pool, hashes)
        sub = R.ladder_subset(ded["kept"], args.n)
        if args.arm == "shuffled":
            sub = R.shuffle_labels(sub, seed=args.seed)
        if args.arm in ("cap", "shuffled"):
            examples = [R.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
            ms = args.max_steps if args.max_steps > 0 else None
            train_rep = R.train_adapter_qa(model, tok, device, examples, lr=args.lr,
                                           epochs=args.epochs, batch_size=args.batch,
                                           warmup_ratio=args.warmup, seed=args.seed,
                                           max_steps=ms,
                                           micro_batch=2)   # math-equivalent accumulation
        else:  # retain: token/step-matched to the SAME-n cap cell, analytically
            cap_examples = [R.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
            budget = sum(len(tok.encode(e["prompt"], add_special_tokens=False)) + 1
                         for e in cap_examples) * args.epochs
            cap_steps = len(R.qa_train_step_batches(cap_examples, args.batch)) * args.epochs
            wt = load_dataset(TE.WIKITEXT_REPO, "wikitext-103-raw-v1", split="train")
            texts = [t for t in wt["text"][:600_000] if t.strip()]
            sl = R.retain_token_matched_slice(texts, tok, budget)
            train_rep = retain_lm_train(model, tok, device, sl["texts"], lr=args.lr,
                                        max_steps=cap_steps,
                                        batch_size=max(2, args.batch // 2),
                                        warmup_ratio=args.warmup, seed=args.seed,
                                        window=256)   # memory envelope: full-vocab CE × r=512 resident (smoke OOM incident)
            train_rep["token_budget_matched"] = budget
            train_rep["retain_slice"] = {"n_tokens": sl["n_tokens"], "n_texts": sl["n_texts"]}
        train_rep["dedup"] = {"n_in": ded["n_in"], "n_dropped": ded["n_dropped"]}

    # ---- eval (one read, the split's filter) ----
    ev = TE.mmlu_choice_logits(model, tok, device, n_per_subject=n_per, item_filter=pred)
    assert ev["per_item_hash"] == split_hashes, f"{cell_id}: eval item order != controls — HALT"
    a_m = ev["per_item"]
    S = s_text(a_m, b_m, f_m)
    Sb = boot_s_text(a_m, b_m, f_m)
    auc = W.rank_auc(ev["choice_logits"], ev["correct_idx"])
    margins = W.margin_per_item(ev["choice_logits"], ev["correct_idx"])
    briers = W.brier_per_item(ev["choice_logits"], ev["correct_idx"])
    spread = float(np.mean([np.std(row) for row in np.asarray(ev["choice_logits"], float)]))

    row = {
        "cell_id": cell_id, "decoder": args.decoder, "arm": args.arm,
        "rank": args.rank, "n": args.n, "seed": args.seed, "eval_split": args.eval_split,
        "adapter": {k: arep[k] for k in ("n_modules", "trainable_params", "rank", "alpha")},
        "train": train_rep,
        "eval": {"acc_pct": ev["acc_pct"], "n_items": ev["n"],
                 "S_recover": S, "S_boot": {"mean": Sb[0], "lo95": Sb[1], "hi95": Sb[2]},
                 "rank_auc": auc, "margin_mean": float(margins.mean()),
                 "brier_mean": float(briers.mean()), "letter_logit_spread": spread},
        "per_item": a_m, "choice_logits": ev["choice_logits"],
        "controls_s3": args.controls_s3, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),   # arch pin audit (prereg §8 MAJOR-7 row)
    }
    (out / f"cell_{cell_id}.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[W17-CELL] {cell_id}: acc={ev['acc_pct']:.2f} S_recover={S:.4f} "
          f"[{Sb[1]:.4f},{Sb[2]:.4f}] auc={auc:.4f} loss={train_rep.get('final_loss')}")


if __name__ == "__main__":
    main()
