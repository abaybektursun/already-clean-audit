"""SAV-A RECIPE SEARCH (prereg §5 HALT branch — the PRE-STATED W17 ladder, on
SAV's own BASE-shell positive control; control-side only).

Candidate B (lr 5e-6, 1 ep, α=r — the W17 winner) failed SAV's fit gate by
0.03–0.04 nats (CE 0.832–0.843 vs ≤ 0.8) while passing no-degradation with
+8.7 pts of headroom. Remaining ladder, same order and selection rule as W17:

    A  lr 1e-5  ep 1  α=r      C  lr 1e-5  ep 1  α=2r
    D  lr 2e-5  ep 1  α=r      E  lr 1e-5  ep 3  α=r
  gates: fit (last-epoch CE ≤ 0.8) AND one-sided no-degradation (BASE bucket ≥
  unprobed − p95 halfwidth). selection: lowest last-epoch CE among passing.

    python -m infra.aws_pg.jobs.sav_recipe_search --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib

CANDIDATES = [
    {"name": "A", "lr": 1e-5, "epochs": 1, "alpha_mult": 1.0},
    {"name": "C", "lr": 1e-5, "epochs": 1, "alpha_mult": 2.0},
    {"name": "D", "lr": 2e-5, "epochs": 1, "alpha_mult": 1.0},
    {"name": "E", "lr": 1e-5, "epochs": 3, "alpha_mult": 1.0},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--fit-ce", type=float, default=0.8)
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
    from infra.pg import w16_probes as W
    from infra.pg.entities import ENTITIES
    from infra.pg.w16_probes import _boot_draws

    assert torch.cuda.is_available()
    device = "cuda"
    print(f"[SAV-RS] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")

    ent = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(ent.repo_id, revision=ent.revision)

    base = AutoModelForCausalLM.from_pretrained(
        ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    base_cl = TE.mmlu_choice_logits(base, tok, device, item_filter=W.is_grid_subset)
    bb = np.asarray(base_cl["per_item"], float)
    draws, _ = _boot_draws(lambda idx: float(bb[idx].mean()), bb.size, 2000, 1337)
    halfwidth = float((np.percentile(draws, 97.5) - np.percentile(draws, 2.5)) / 2)
    floor_ref = float(bb.mean() - halfwidth)
    print(f"[SAV-RS] BASE bucket={bb.mean():.4f} no-degradation floor={floor_ref:.4f}")
    del base
    torch.cuda.empty_cache()

    # dedup needs the FULL eval-population hashes; compute them from the test
    # split directly (no forward needed — the hash is a function of the question)
    import hashlib as _hl
    ds_all = load_dataset(TE.MMLU_REPO, "all")
    eval_hashes = [_hl.sha1(r["question"][:64].encode()).hexdigest()[:12]
                   for r in ds_all["test"]]
    pool = [dict(r) for r in ds_all["auxiliary_train"]]
    ded = R.dedup_against_eval(pool, eval_hashes)
    ex16 = [R.format_qa(r["question"], r["choices"], r["answer"])
            for r in R.ladder_subset(ded["kept"], args.n)]

    rows = []
    for cand in CANDIDATES:
        m = AutoModelForCausalLM.from_pretrained(
            ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
            device_map=device, attn_implementation="eager").eval()
        R.attach_adapters(m.model, rank=512, seed=1337, alpha=cand["alpha_mult"] * 512)
        m.to(device)
        tr = R.train_adapter_qa(m, tok, device, ex16, lr=cand["lr"],
                                epochs=cand["epochs"], batch_size=args.batch,
                                warmup_ratio=args.warmup, seed=1337, micro_batch=2)
        ev = TE.mmlu_choice_logits(m, tok, device, item_filter=W.is_grid_subset)
        fit_ok = bool(tr["mean_loss_last_epoch"] <= args.fit_ce)
        nodeg_ok = bool(ev["acc"] >= floor_ref)
        row = {**cand, "last_epoch_ce": tr["mean_loss_last_epoch"],
               "bucket_acc_pct": ev["acc_pct"], "fit_ok": fit_ok,
               "no_degradation_ok": nodeg_ok, "pass": bool(fit_ok and nodeg_ok),
               "steps": tr["steps"]}
        rows.append(row)
        print(f"[SAV-RS] {cand['name']}: lr={cand['lr']:g} ep={cand['epochs']} "
              f"α={cand['alpha_mult']:g}r → CE {row['last_epoch_ce']:.3f} "
              f"bucket {row['bucket_acc_pct']:.2f} fit={fit_ok} nodeg={nodeg_ok}")
        del m
        torch.cuda.empty_cache()

    passing = [r for r in rows if r["pass"]]
    winner = min(passing, key=lambda r: r["last_epoch_ce"]) if passing else None
    result = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "floor_ref": floor_ref, "halfwidth": halfwidth,
              "failed_candidate_B": {"lr": 5e-6, "epochs": 1, "alpha_mult": 1.0,
                                     "ce": [0.832, 0.843, 0.834], "nodeg": True,
                                     "source": "results/sav_controls_a1"},
              "candidates": rows, "winner": winner,
              "selection_rule": "lowest last-epoch CE among candidates passing BOTH gates"}
    (out / "recipe_search.json").write_text(json.dumps(result, indent=1))
    print(f"[SAV-RS] WINNER: {winner}")
    if winner is None:
        raise RuntimeError("SAV RECIPE SEARCH: no candidate passes — fit-form amendment "
                           "required (disclosed, pre-π-side)")


if __name__ == "__main__":
    main()
