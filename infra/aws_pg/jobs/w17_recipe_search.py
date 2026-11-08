"""W17 RECIPE SEARCH (the prereg §5 HALT branch — documented, control-side ONLY).

The eq1-candidate recipe (lr 2e-5, ep 3, α=2r) failed the positive-control
gates at controls a2: it fits (CE→0.17) but DEGRADES the capability-intact
BEFORE side by 8–16 bucket points (overfit/distortion at composite scale 2.0,
6,000 steps). Per the frozen rule: HALT + documented search. This job runs a
PRE-STATED candidate ladder on the BEFORE-side positive control (cap-data
r=512 × n=16k, adapter seed 1337, grid-bucket eval — no AFTER-side number is
touched) and applies a PRE-STATED selection rule:

  candidates (in order):
    A  lr 1e-5  ep 1  α=r      B  lr 5e-6  ep 1  α=r
    C  lr 1e-5  ep 1  α=2r     D  lr 2e-5  ep 1  α=r
    E  lr 1e-5  ep 3  α=r
  gates per candidate: fit (last-epoch letter-CE ≤ 0.5) AND one-sided
    no-degradation (bucket acc ≥ unprobed-bucket − p95 halfwidth, recomputed
    here from the same controls-a2 arrays construction).
  selection: among candidates passing BOTH, the LOWEST last-epoch CE (best
    fit). If none pass both: report all rows; the fit-form mis-specification
    is then amended in the prereg BEFORE any AFTER-side run (disclosed).

    python -m infra.aws_pg.jobs.w17_recipe_search --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

W16_CONTROLS_S3 = "s3://pg-lineage-769063704696/results/w16_controls_a1/out/w16_controls.json"
CANDIDATES = [
    {"name": "A", "lr": 1e-5, "epochs": 1, "alpha_mult": 1.0},
    {"name": "B", "lr": 5e-6, "epochs": 1, "alpha_mult": 1.0},
    {"name": "C", "lr": 1e-5, "epochs": 1, "alpha_mult": 2.0},
    {"name": "D", "lr": 2e-5, "epochs": 1, "alpha_mult": 1.0},
    {"name": "E", "lr": 1e-5, "epochs": 3, "alpha_mult": 1.0},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--w16-controls-s3", default=W16_CONTROLS_S3)
    ap.add_argument("--n", type=int, default=16_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
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
    from transformers import AutoTokenizer

    from infra.pg import relearn as R
    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as W
    from infra.pg.entities import ENTITIES
    from infra.pg.w16_probes import _boot_draws

    assert torch.cuda.is_available()
    device = "cuda"
    print(f"[W17-RS] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")

    w16_local = out / "w16_controls_cached.json"
    subprocess.run(["aws", "s3", "cp", args.w16_controls_s3, str(w16_local),
                    "--region", "us-east-1"], check=True)
    W16C = json.loads(w16_local.read_text())
    hashes = W16C["item_hashes"]
    b_bucket = W.mask_by(W16C["BEFORE"]["per_item"], hashes, W.is_grid_subset)
    bb = np.asarray(b_bucket, float)
    draws, _ = _boot_draws(lambda idx: float(bb[idx].mean()), bb.size, 2000, 1337)
    halfwidth = float((np.percentile(draws, 97.5) - np.percentile(draws, 2.5)) / 2)
    floor_ref = float(bb.mean() - halfwidth)
    print(f"[W17-RS] no-degradation floor = {floor_ref:.4f} (unprobed {bb.mean():.4f})")

    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    ds = load_dataset(TE.MMLU_REPO, "all")
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ded = R.dedup_against_eval(pool, hashes)
    ex16 = [R.format_qa(r["question"], r["choices"], r["answer"])
            for r in R.ladder_subset(ded["kept"], args.n)]

    rows = []
    for cand in CANDIDATES:
        _pa, pg_before, _p, paths, _i, _g = H.build_before_after(device)
        del _pa
        torch.cuda.empty_cache()
        lm = pg_before.get_submodule(paths["language_model"])
        R.attach_adapters(lm, rank=512, seed=1337, alpha=cand["alpha_mult"] * 512)
        pg_before.to(device)
        tr = R.train_adapter_qa(pg_before, tok, device, ex16, lr=cand["lr"],
                                epochs=cand["epochs"], batch_size=args.batch,
                                warmup_ratio=args.warmup, seed=1337, micro_batch=2)
        ev = TE.mmlu_choice_logits(pg_before, tok, device, item_filter=W.is_grid_subset)
        fit_ok = bool(tr["mean_loss_last_epoch"] <= 0.5)
        nodeg_ok = bool(ev["acc"] >= floor_ref)
        row = {**cand, "last_epoch_ce": tr["mean_loss_last_epoch"],
               "first_epoch_ce": tr["mean_loss_first_epoch"],
               "bucket_acc_pct": ev["acc_pct"], "fit_ok": fit_ok,
               "no_degradation_ok": nodeg_ok, "pass": bool(fit_ok and nodeg_ok),
               "steps": tr["steps"]}
        rows.append(row)
        print(f"[W17-RS] {cand['name']}: lr={cand['lr']:g} ep={cand['epochs']} "
              f"α={cand['alpha_mult']:g}r → CE {row['first_epoch_ce']:.3f}→{row['last_epoch_ce']:.3f} "
              f"bucket {row['bucket_acc_pct']:.2f} fit={fit_ok} nodeg={nodeg_ok}")
        del pg_before
        torch.cuda.empty_cache()

    passing = [r for r in rows if r["pass"]]
    winner = min(passing, key=lambda r: r["last_epoch_ce"]) if passing else None
    result = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "floor_ref": floor_ref, "halfwidth": halfwidth,
              "candidates": rows, "winner": winner,
              "selection_rule": "lowest last-epoch CE among candidates passing BOTH gates"}
    (out / "recipe_search.json").write_text(json.dumps(result, indent=1))
    print(f"[W17-RS] WINNER: {winner}")
    if winner is None:
        raise RuntimeError("RECIPE SEARCH: no candidate passes both gates — "
                           "fit-form amendment required (disclosed, pre-AFTER)")


if __name__ == "__main__":
    main()
