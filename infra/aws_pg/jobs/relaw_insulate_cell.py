"""RELAW Arm-3 insulation cell — ONE {mode, seed} consuming-FT → re-entry-RF cell (prereg §3.4).

The causal/practical lever. Start from the LATMECH `deleted` PG2 model (MMLU MCQ deleted but
LATENT, re-learnable to ceiling), fine-tune the FULL 2-D decoder on a generic consuming task
(wikitext LM) ONE of three ways, then measure re-entry RF of the deleted MMLU capability:

  mode=standard : raw decoder-weight gradient (the natural FT).
  mode=off      : INSULATED — each step the decoder gradient is orthogonalised OFF the top-σ
                  principal subspace of the start model (relaw.insulated_update).
  mode=into     : ANTI-insulated — the in-subspace component, MAGNITUDE-MATCHED to the insulated
                  step, pointed INTO the subspace (the adversarial null; Arm-3 denominator).

After the consuming FT: attach rank-r adapters, relearn the MMLU ladder, co-measure MCQ (blank
image, paired per item) — the SAME G instrument as latmech_cell. RF vs latmech floor/ceiling is
computed downstream (relaw_insulate_unblind). Fail-loud; the cell is the re-run unit.

    python -m infra.aws_pg.jobs.relaw_insulate_cell --out <dir> --mode off --seed 1337 \
        --insulate-rank 64 --consume-steps 300 --consume-lr 1e-4 --consume-batch 2 \
        --rank 512 --n 16000 --eval-split confirmation --lr 5e-6 --epochs 1 --batch 8 --warmup 0.03
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True, choices=["standard", "off", "into"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--insulate-rank", type=int, default=64)
    ap.add_argument("--consume-steps", type=int, default=300)
    ap.add_argument("--consume-lr", type=float, default=1e-4)
    ap.add_argument("--consume-batch", type=int, default=2)
    ap.add_argument("--consume-seqlen", type=int, default=256)
    ap.add_argument("--consume-momentum", type=float, default=0.9)
    # re-entry (G) instrument — the frozen relearn recipe (latmech_cell verbatim).
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--eval-split", choices=["bucket", "confirmation"], required=True)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--n-per-subject", type=int, default=0, help="smoke cap (0 = full split)")
    args = ap.parse_args()

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
    from infra.pg import relaw as RL
    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as W
    from infra.pg import w14_localize as W14
    from infra.pg import latmech as Lm
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "relaw_insulate_cell needs GPU."
    device = "cuda"
    n_per = args.n_per_subject or None
    cell_id = f"insulate_{args.mode}_r{args.insulate_rank}_n{args.n}_s{args.seed}_{args.eval_split}"
    print(f"[RELAW-INSULATE] {cell_id} consume(steps={args.consume_steps} lr={args.consume_lr} "
          f"b={args.consume_batch}) relearn(r={args.rank} lr={args.lr})", flush=True)

    pred = W.is_grid_subset if args.eval_split == "bucket" else W.is_confirmation
    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    dkeys = W14.decoder_keys()

    # ---- start model: the LATMECH `deleted` PG2 decoder (MMLU latent) ----
    pg_after, pg_before, proc, paths, _it, _gate = H.build_before_after(device)
    del pg_before
    torch.cuda.empty_cache()
    model = pg_after
    lm = model.get_submodule(paths["language_model"])
    nlay = RL.n_decoder_layers(dkeys)
    assert nlay == 26, f"generation firewall: Arm-3 is Gemma-2 (26L), got {nlay}"

    # ---- cache the top-r σ-subspace of the START model's decoder (the scaffold to (de)protect) ----
    subspace = {}
    for k in dkeys:
        Wk = lm.get_parameter(k).detach().float().to(device)
        Ur, Vr = RL.topk_subspace(Wk, args.insulate_rank)
        subspace[k] = (Ur.to(device).to(torch.bfloat16), Vr.to(device).to(torch.bfloat16))
    print(f"[RELAW-INSULATE] cached top-{args.insulate_rank} subspace for {len(subspace)} matrices", flush=True)

    # ---- consuming FT: full 2-D decoder, SGD+momentum, per-step insulation surgery ----
    for p in model.parameters():
        p.requires_grad_(False)
    train_params = []
    for k in dkeys:
        prm = lm.get_parameter(k)
        prm.requires_grad_(True)
        train_params.append((k, prm))
    # NO gradient checkpointing: full-decoder FT at small batch/seq fits the L40S (~15GB) without it,
    # and reentrant checkpointing errors when no forward INPUT requires grad (only the weights do).
    model.train()

    wiki = [t for t in load_dataset(TE.WIKITEXT_REPO, TE.WIKITEXT_CONFIG)["train"]["text"]
            if len(t.strip()) >= 200]
    g_data = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(wiki), generator=g_data).tolist()
    opt = torch.optim.SGD([p for _, p in train_params], lr=args.consume_lr,
                          momentum=args.consume_momentum)

    upd_norm_in, upd_norm_g = 0.0, 0.0
    di = 0
    consume_losses = []
    for step in range(args.consume_steps):
        texts = []
        while len(texts) < args.consume_batch:
            texts.append(wiki[order[di % len(order)]])
            di += 1
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.consume_seqlen).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        loss = model(**enc, labels=labels).loss
        assert torch.isfinite(loss), f"non-finite consuming loss at step {step} — HALT"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # per-matrix insulation surgery (in place, before the optimizer step).
        with torch.no_grad():
            for k, prm in train_params:
                if prm.grad is None:
                    continue
                Ur, Vr = subspace[k]
                gnew = RL.insulated_update(prm.grad, Ur, Vr, args.mode)
                if step == 0:
                    upd_norm_g += float(prm.grad.float().norm() ** 2)
                    upd_norm_in += float(gnew.float().norm() ** 2)
                prm.grad.copy_(gnew.to(prm.grad.dtype))
        opt.step()
        consume_losses.append(float(loss.item()))
        if step % 50 == 0:
            print(f"[RELAW-INSULATE] consume step {step}: loss={loss.item():.4f}", flush=True)
    model.eval()
    print(f"[RELAW-INSULATE] consume done: loss {consume_losses[0]:.3f}→{consume_losses[-1]:.3f} | "
          f"step0 ‖raw_g‖={upd_norm_g**0.5:.2f} ‖applied‖={upd_norm_in**0.5:.2f}", flush=True)
    del subspace
    torch.cuda.empty_cache()

    # ---- re-entry G instrument: attach adapters, relearn MMLU, co-measure MCQ (latmech_cell spine) ----
    for p in model.parameters():
        p.requires_grad_(False)
    arep = Rl.attach_adapters(lm, rank=args.rank, seed=args.seed, alpha=float(args.rank))
    model.to(device)
    ds = load_dataset(TE.MMLU_REPO, "all")
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ev_hashes = [Rl.qa_item_hash(r["question"]) for r in ds["test"]]
    ded = Rl.dedup_against_eval(pool, ev_hashes)
    sub = Rl.ladder_subset(ded["kept"], args.n)
    examples = [Rl.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
    train_rep = Rl.train_adapter_qa(model, tok, device, examples, lr=args.lr, epochs=args.epochs,
                                    batch_size=args.batch, warmup_ratio=args.warmup,
                                    seed=args.seed, micro_batch=2)

    mcq = TE.mmlu_blankimg(model, proc, device, n_per_subject=n_per, item_filter=pred)
    clz = TE.mmlu_cloze(model, tok, device, processor=proc, n_per_subject=n_per, item_filter=pred)
    Lm.position_align_tripwire(mcq["per_item_hash"], clz["per_item_hash"])
    mcq_item = mcq["per_item"]

    row = {
        "cell_id": cell_id, "arm": "insulate", "mode": args.mode,
        "insulate_rank": args.insulate_rank, "rank": args.rank, "n": args.n, "seed": args.seed,
        "eval_split": args.eval_split,
        "consume": {"steps": args.consume_steps, "lr": args.consume_lr, "batch": args.consume_batch,
                    "seqlen": args.consume_seqlen, "momentum": args.consume_momentum,
                    "loss_first": consume_losses[0], "loss_last": consume_losses[-1],
                    "step0_raw_grad_norm": upd_norm_g ** 0.5, "step0_applied_norm": upd_norm_in ** 0.5},
        "adapter": {k: arep[k] for k in ("n_modules", "trainable_params", "rank", "alpha")},
        "train": train_rep,
        "item_hash": clz["per_item_hash"], "correct_idx": clz["correct_idx"],
        "mcq": {"acc_pct": round(100 * sum(mcq_item) / len(mcq_item), 2),
                "n_items": len(mcq_item), "per_item": mcq_item},
        "cloze": {"acc_pct": clz["acc_pct"], "n_items": clz["n"], "per_item": clz["per_item"]},
        "recipe": {"lr": args.lr, "epochs": args.epochs, "batch": args.batch, "warmup": args.warmup},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    (out / f"cell_{cell_id}.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[RELAW-INSULATE] {cell_id}: MCQ={row['mcq']['acc_pct']:.2f} cloze={clz['acc_pct']:.2f} "
          f"n={clz['n']} relearn_loss={train_rep.get('final_loss')}", flush=True)
    print(f"RELAW_INSULATE_OK mode={args.mode} mcq={row['mcq']['acc_pct']:.2f}", flush=True)


if __name__ == "__main__":
    main()
