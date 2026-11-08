"""TRAINSCAFFOLD Arm-2 REACHABILITY — full-FT MMLU re-entry: refill vs refill-BLOCKED (prereg §3.3).

Full-decoder fine-tune (the substrate's 2-D decoder weights) on the MMLU re-acquisition data (single-
letter CE, the relearn target), then measure RF_MCQ. mode=standard = unconstrained REFILL (can rebuild
the zeroed directions — recovery is expected by construction, the pre-registered null); mode=off =
refill-BLOCKED (each step the decoder-weight gradient is projected OFF the top-σ block of the removed
carrier — relaw.project_grad — so the FT cannot rebuild the removed structure; THIS tests severance).
substrate ablated (A) or deleted (the non-ablated full-FT positive control). The route-around rank
ladder (rank 512/1024/2048) is the REUSED latmech_cell at higher rank (run separately). Fail-loud.

    python -m infra.aws_pg.jobs.trainscaffold_reach --out <dir> --substrate ablated --mode off \
        --carrier s3://.../topsig/carrier.pt --consume-steps 2000 --lr 1e-4 --eval-split confirmation
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--substrate", required=True, choices=["ablated", "deleted"])
    ap.add_argument("--mode", required=True, choices=["standard", "off"], help="off = refill-BLOCKED")
    ap.add_argument("--carrier", default="", help="topsig carrier.pt (required for ablated / off)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--n-train", type=int, default=16000)
    ap.add_argument("--eval-split", choices=["bucket", "confirmation"], default="confirmation")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-per-subject", type=int, default=0)
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
    from infra.pg import text_eval as TE
    from infra.pg import w14_localize as W14
    from infra.pg import w16_probes as W
    from infra.pg import relaw as RL
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs import _w15_shared as S15

    assert torch.cuda.is_available(), "trainscaffold_reach needs GPU."
    device = "cuda"
    n_per = args.n_per_subject or None
    dkeys = W14.decoder_keys()
    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    pred = W.is_grid_subset if args.eval_split == "bucket" else W.is_confirmation
    cid = f"reach_{args.substrate}_{args.mode}_s{args.seed}"
    print(f"[TS-REACH] {cid} steps={args.steps} lr={args.lr}", flush=True)

    # ---- substrate + (for off / ablated) the removed top-σ subspace from the carrier ----
    factors = None
    if args.carrier:
        cpath = args.carrier
        if cpath.startswith("s3://"):
            import subprocess
            cpath = str(out / "carrier.pt")
            subprocess.run(["aws", "s3", "cp", args.carrier, cpath, "--region", "us-east-1"], check=True)
        factors = torch.load(cpath, map_location="cpu")

    pg_after, pg_before, proc, paths, _it, _gate = H.build_before_after(device)
    del pg_before
    model = pg_after
    lm = model.get_submodule(paths["language_model"])
    if args.substrate == "ablated":
        assert factors is not None, "ablated requires --carrier"
        base_sd = {k: v.detach() for k, v in lm.state_dict().items()}
        S15.apply_carrier(lm, base_sd, dkeys, factors, "carrier", device)
    torch.cuda.empty_cache()

    # cache the removed top-σ left/right subspaces (for the refill-BLOCKED projection)
    subspace = {}
    if args.mode == "off":
        assert factors is not None, "refill-blocked (off) requires --carrier"
        for k in dkeys:
            Uc, _Sc, Vc = factors[k]["carrier"]
            subspace[k] = (Uc.to(device).to(torch.bfloat16), Vc.to(device).to(torch.bfloat16))

    # ---- full-decoder FT on MMLU letter-CE ----
    for p in model.parameters():
        p.requires_grad_(False)
    train_params = []
    for k in dkeys:
        prm = lm.get_parameter(k)
        prm.requires_grad_(True)
        train_params.append((k, prm))
    model.train()

    ds = load_dataset(TE.MMLU_REPO, "all")
    poolrows = [dict(r) for r in ds["auxiliary_train"]]
    ev_hashes = [Rl.qa_item_hash(r["question"]) for r in ds["test"]]
    ded = Rl.dedup_against_eval(poolrows, ev_hashes)
    sub = Rl.ladder_subset(ded["kept"], args.n_train)
    ex = [Rl.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
    letters = Rl.letter_token_ids(tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    opt = torch.optim.SGD([p for _, p in train_params], lr=args.lr, momentum=args.momentum)
    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(ex), generator=g).tolist()
    di, losses = 0, []
    for step in range(args.steps):
        idx = [order[(di + j) % len(order)] for j in range(args.batch)]
        di += args.batch
        prompts = [ex[i]["prompt"] for i in idx]
        tgt = torch.tensor([letters[ex[i]["answer_idx"]] for i in idx], device=device)
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        pos = enc["attention_mask"].long().cumsum(-1) - 1
        pos = pos.masked_fill(enc["attention_mask"] == 0, 1)
        last = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :]
        loss = torch.nn.functional.cross_entropy(last.float(), tgt)
        assert torch.isfinite(loss), f"non-finite loss step {step} — HALT"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.mode == "off":                       # refill-BLOCKED: project grad off the removed block
            with torch.no_grad():
                for k, prm in train_params:
                    if prm.grad is None:
                        continue
                    Ur, Vr = subspace[k]
                    prm.grad.copy_(RL.project_grad(prm.grad, Ur, Vr, "off").to(prm.grad.dtype))
        opt.step()
        losses.append(float(loss.item()))
        if step % 200 == 0:
            print(f"[TS-REACH] step {step}: loss={loss.item():.4f}", flush=True)
    model.eval()

    mcq = TE.mmlu_blankimg(model, proc, device, n_per_subject=n_per, item_filter=pred)
    row = {"cell_id": cid, "substrate": args.substrate, "mode": args.mode, "seed": args.seed,
           "steps": args.steps, "lr": args.lr, "eval_split": args.eval_split,
           "loss_first": losses[0], "loss_last": losses[-1],
           "mcq": {"acc_pct": round(100 * sum(mcq["per_item"]) / len(mcq["per_item"]), 2),
                   "n_items": len(mcq["per_item"]), "per_item": mcq["per_item"]},
           "item_hash": mcq["per_item_hash"], "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0)}
    (out / f"cell_{cid}.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[TS-REACH] {cid}: MCQ={row['mcq']['acc_pct']:.2f} loss {losses[0]:.3f}->{losses[-1]:.3f}", flush=True)
    print(f"TS_REACH_OK {cid} mcq={row['mcq']['acc_pct']:.2f}", flush=True)


if __name__ == "__main__":
    main()
