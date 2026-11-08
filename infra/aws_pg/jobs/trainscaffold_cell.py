"""TRAINSCAFFOLD novel-task cell — ONE {substrate, format, shuffle, seed} cell (prereg §3.1/§3.2).

Build the backbone substrate (HEALTHY / DELETED=B / ABLATED=A / RANDOM=R), attach a fresh rank-r
adapter, train it on the invented-knowledge battery in one FORMAT (free-gen masked-LM or letter-CE),
then read held-out RECOGNITION (the readout-agnostic cloze probe — the same fixed probe set across all
substrates so per-item arrays are PAIRED for the joint bootstrap). `--shuffle` trains the permuted-
association twin (must fail the real probe). RF_novel + the verdict are computed downstream
(trainscaffold_unblind). Fail-loud; the cell is the re-run unit.

    python -m infra.aws_pg.jobs.trainscaffold_cell --out <dir> --substrate ablated --format free \
        --seed 1337 --carrier s3://.../latmech-topsig/out/topsig/carrier.pt \
        --n-train 16000 --n-eval 2000 --rank 512 --lr 5e-6 --epochs 1 --batch 8 --warmup 0.03
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--substrate", required=True, choices=["healthy", "deleted", "ablated", "random"])
    ap.add_argument("--format", required=True, choices=["free", "letter"])
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--carrier", default="", help="topsig carrier.pt (ablated; local or s3://)")
    ap.add_argument("--emrand", action="store_true", help="ablated: use the emrand matched-null factors")
    ap.add_argument("--facts-seed", type=int, default=1337)
    ap.add_argument("--n-entities", type=int, default=4000)
    ap.add_argument("--n-train", type=int, default=16000)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--decoder-seed", type=int, default=1337)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch
    from transformers import AutoTokenizer

    from infra.pg import relearn as Rl
    from infra.pg import w14_localize as W14
    from infra.pg import trainscaffold as TS  # noqa: F401 (parity; verdict downstream)
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs import _w15_shared as S15
    from infra.aws_pg.jobs.trainscaffold_facts import build_battery
    from infra.aws_pg.jobs.trainscaffold_train import train_adapter_lm, recognition_cloze

    assert torch.cuda.is_available(), "trainscaffold_cell needs GPU."
    device = "cuda"
    cid = f"{args.substrate}_{args.format}{'_shuf' if args.shuffle else ''}_r{args.rank}_s{args.seed}"
    print(f"[TS-CELL] {cid} train={args.n_train} eval={args.n_eval}", flush=True)

    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    dkeys = W14.decoder_keys()

    # ---- battery (deterministic) + the FIXED eval-probe set (paired across substrates) ----
    bat = build_battery(args.n_entities, seed=args.facts_seed)
    facts = bat["facts"][:args.n_train]
    eval_facts = bat["facts"][:args.n_eval]                       # first n_eval by fid order = paired
    shuf = bat["shuffle"]

    # ---- build the substrate (latmech_cell pattern) ----
    pg_after, pg_before, proc, paths, _it, _gate = H.build_before_after(device)
    if args.substrate == "random":
        model = H.build_random_after(pg_before, paths, device, seed=args.decoder_seed)
        del pg_after, pg_before
    elif args.substrate == "healthy":
        model = pg_before
        del pg_after
    elif args.substrate == "ablated":
        assert args.carrier, "ablated requires --carrier (topsig carrier.pt)"
        cpath = args.carrier
        if cpath.startswith("s3://"):
            import subprocess
            cpath = str(out / "carrier.pt")
            subprocess.run(["aws", "s3", "cp", args.carrier, cpath, "--region", "us-east-1"], check=True)
        factors = torch.load(cpath, map_location="cpu")
        model = pg_after
        lm = model.get_submodule(paths["language_model"])
        base_sd = {k: v.detach() for k, v in lm.state_dict().items()}
        which = "emrand" if args.emrand else "carrier"
        n_ov = S15.apply_carrier(lm, base_sd, dkeys, factors, which, device)
        print(f"[TS-CELL] ablated: applied {which} to {n_ov} matrices", flush=True)
        del pg_before
    else:  # deleted (B)
        model = pg_after
        del pg_before
    torch.cuda.empty_cache()
    lm = model.get_submodule(paths["language_model"])

    # ---- adapters + train (format-matched recipe) ----
    arep = Rl.attach_adapters(lm, rank=args.rank, seed=args.seed, alpha=float(args.rank))
    model.to(device)
    src = (lambda f: shuf[f["fid"]]) if args.shuffle else (lambda f: f)
    if args.format == "free":
        examples = [{"prompt": src(f)["free"]["prompt"], "target": src(f)["free"]["target"]} for f in facts]
        train_rep = train_adapter_lm(model, tok, device, examples, lr=args.lr, epochs=args.epochs,
                                     batch_size=args.batch, warmup_ratio=args.warmup,
                                     seed=args.seed, micro_batch=2)
    else:
        examples = [{"prompt": src(f)["letter_mc"]["prompt"],
                     "answer_idx": src(f)["letter_mc"]["answer_idx"]} for f in facts]
        train_rep = Rl.train_adapter_qa(model, tok, device, examples, lr=args.lr, epochs=args.epochs,
                                        batch_size=args.batch, warmup_ratio=args.warmup,
                                        seed=args.seed, micro_batch=2)

    # ---- held-out RECOGNITION (the fixed probe set; real facts' probes; paired by fid) ----
    probe_items = [{"prompt": f["probe"]["prompt"], "candidates": f["probe"]["candidates"],
                    "answer_idx": f["probe"]["answer_idx"], "fid": f["fid"]} for f in eval_facts]
    rec = recognition_cloze(model, proc, device, probe_items, batch_size=8)

    row = {
        "cell_id": cid, "substrate": args.substrate, "format": args.format, "shuffle": args.shuffle,
        "emrand": args.emrand, "seed": args.seed, "rank": args.rank,
        "battery_hash": bat["hash"], "n_train": len(facts), "n_eval": len(eval_facts),
        "adapter": {k: arep[k] for k in ("n_modules", "trainable_params", "rank", "alpha")},
        "train": {k: train_rep[k] for k in ("steps", "final_loss", "mean_loss_first_epoch",
                                            "mean_loss_last_epoch", "loss_trajectory", "recipe")},
        "recognition": {"acc": rec["acc"], "n": rec["n"], "per_item": rec["per_item"],
                        "fids": [p["fid"] for p in probe_items]},
        "recipe": {"lr": args.lr, "epochs": args.epochs, "batch": args.batch, "warmup": args.warmup},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    (out / f"cell_{cid}.json").write_text(json.dumps(row, indent=1, default=str))
    print(f"[TS-CELL] {cid}: recognition acc={rec['acc']:.4f} (n={rec['n']}) "
          f"train_loss {train_rep['mean_loss_first_epoch']:.3f}->{train_rep['mean_loss_last_epoch']:.3f}",
          flush=True)
    print(f"TS_CELL_OK {cid} acc={rec['acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
