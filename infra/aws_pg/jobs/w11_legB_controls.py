"""W11 §9 Leg-B CONTROLS (step 2/4: develop on the DISCOVERY half; instrument-works gate).

Load PG2-mix-224 (the task-ready checkpoint — the only way to measure ablation BREAKAGE without
training; the connector architecture is shared with pt, scoping stated). SVD the multi_modal
connector W_proj, harvest h̄ on discovery images, then on the DISCOVERY eval items per task:

  baseline      no ablation                                   → the per-task reference
  all-ablate    Q = U (full rank) → projector DEAD            → positive control: must bite
  N4 shuffle    roll projector output across batch            → must collapse (image↔text broken)
  random-floor  k random orthonormal dirs (sanity k × 3 seeds) → the floor structured must beat

Gates (instrument-works, controls-beat-null):
  all_ablate_ok : median per-item (baseline − all-ablate) ≥ 0.20 → ablation has bite.
  n4_ok         : median per-item (baseline − N4) ≥ 0.15 → the bridge output carries the image.
  random_ok     : random-ablation scores finite and ≤ baseline.

FREEZES the SVD (U,S) + h̄ + the k-grid to projector_svd.pt (reused bit-identical at unblind).
The structured-vs-random concentration DECISION is read only at unblind on the SEALED confirmation
half. Fail-loud; no try/except.

    python -m infra.aws_pg.jobs.w11_legB_controls --out <dir>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys

N_TOTAL = 640          # per task; ~320 discovery / ~320 confirmation
SPLIT_SEED = 1337
LOAD_SEED = 0
CTRL_RAND_K = (8, 128)
BATCH = 16
POOL_IMGS_PER_TASK = 64


def _ensure_pycocoevalcap():
    if importlib.util.find_spec("pycocoevalcap") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocoevalcap"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-total", type=int, default=N_TOTAL)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    _ensure_pycocoevalcap()

    import numpy as np
    import torch
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    from infra.pg import projector_ablation as PA
    from infra.pg.entities import ENTITIES, resolve_submodule_paths

    assert torch.cuda.is_available(), "Leg-B controls need GPU."
    device = "cuda"
    print(f"[W11-B-CTRL] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")

    mix_repo = ENTITIES["PG_mix"].repo_id
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        mix_repo, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa").eval()
    proc = PaliGemmaProcessor.from_pretrained(mix_repo)
    paths = resolve_submodule_paths(model)

    proj_module, W, b = PA.get_projector(model, paths)
    U, S = PA.svd_U(W)                                   # U:(out,r) S:(r)
    U = U.to(device).to(torch.float32)
    out_dim, r = U.shape
    print(f"[W11-B-CTRL] W_proj shape={tuple(W.shape)} rank r={r} out_dim={out_dim} "
          f"S[:5]={[round(float(x),3) for x in S[:5]]}")

    tasks = PA.build_tasks()
    loaded = {}
    pooled = []
    item_hashes = {}
    splits = {}
    for task in tasks:
        rows = PA.load_eval_items(task, args.n_total, seed=LOAD_SEED)
        sp = PA.split_items(len(rows), seed=SPLIT_SEED)
        disc_rows = [rows[i] for i in sp["disc_idx"]]
        loaded[task.name] = (rows, sp, disc_rows)
        item_hashes[task.name] = [PA._content_hash(task, r) for r in rows]
        splits[task.name] = {"disc_idx": [int(i) for i in sp["disc_idx"]],
                             "conf_idx": [int(i) for i in sp["conf_idx"]]}
        for i in sp["disc_idx"][:POOL_IMGS_PER_TASK]:
            pooled.append(task.image_fn(rows[i]))
        print(f"[W11-B-CTRL] {task.name}: loaded {len(rows)} (disc {len(sp['disc_idx'])}/"
              f"conf {len(sp['conf_idx'])})")

    hbar = PA.harvest_hbar(model, proc, pooled[:256], proj_module, device, batch_size=BATCH)
    print(f"[W11-B-CTRL] h̄ harvested over {min(len(pooled),256)} images, |h̄|={float(hbar.norm()):.3f}")

    # pt↔mix projector drift (scoping disclosure: this Leg characterizes the MIX connector; how far
    # is it from the pt lineage subject's connector?). Load pt projector on CPU, compare Frobenius.
    pt_repo = ENTITIES["PG"].repo_id
    pt_model = PaliGemmaForConditionalGeneration.from_pretrained(
        pt_repo, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa")
    _, W_pt, _ = PA.get_projector(pt_model, resolve_submodule_paths(pt_model))
    proj_pt_mix_rel_frob = float((W - W_pt).norm() / W_pt.norm())
    del pt_model
    print(f"[W11-B-CTRL] projector pt→mix rel-Frobenius drift={proj_pt_mix_rel_frob:.4f}")

    per_task = {}
    for task in tasks:
        _, _, disc_rows = loaded[task.name]
        base = PA.eval_task(model, proc, task, disc_rows, device, batch_size=BATCH, cider_boot=200)
        allab = PA.eval_task(model, proc, task, disc_rows, device,
                             hook_fn=PA.make_ablation_hook(U, hbar), proj_module=proj_module,
                             batch_size=BATCH, cider_boot=200)
        n4 = PA.eval_task(model, proc, task, disc_rows, device,
                          hook_fn=PA.make_n4_shuffle_hook(), proj_module=proj_module,
                          batch_size=BATCH, cider_boot=200)
        rand = {}
        for k in CTRL_RAND_K:
            for s in range(3):
                Q = PA.random_basis(out_dim, k, seed=1000 * k + s, device=device, dtype=torch.float32)
                rr = PA.eval_task(model, proc, task, disc_rows, device,
                                  hook_fn=PA.make_ablation_hook(Q, hbar), proj_module=proj_module,
                                  batch_size=BATCH, cider_boot=120)
                rand[f"k{k}_s{s}"] = rr["score"]
        # NB: structured top-k ablation is NOT run on discovery — it is the sealed DECISION,
        # read only at unblind on the confirmation half (blind discipline; critic fix).
        per_task[task.name] = {
            "corpus": base["corpus"], "n_disc": base["n"],
            "baseline": base["score"], "baseline_ci95": base["ci95"],
            "all_ablate": allab["score"], "all_ablate_ci95": allab["ci95"],
            "n4_shuffle": n4["score"], "random_floor": rand,
        }
        print(f"[W11-B-CTRL] {task.name}: base={base['score']:.4f} all-ablate={allab['score']:.4f} "
              f"N4={n4['score']:.4f} rand_k8≈{rand.get('k8_s0'):.4f}")

    per_item_tasks = [t for t in per_task if not per_task[t]["corpus"]]
    all_drop = [per_task[t]["baseline"] - per_task[t]["all_ablate"] for t in per_item_tasks]
    n4_drop = [per_task[t]["baseline"] - per_task[t]["n4_shuffle"] for t in per_item_tasks]
    all_ablate_ok = bool(np.median(all_drop) >= 0.20 and all(d > 0 for d in all_drop))
    n4_ok = bool(np.median(n4_drop) >= 0.15)
    # random ablation must be a WEAK floor: NEAR baseline (within a noise band), not strictly below
    # — noise legitimately nudges it above baseline, and that IS the expected near-inert behavior
    # (all-ablate drops of 0.4-0.63 sit far outside this band → clean separation). §9-4 recalibration.
    RAND_BAND = 0.05
    rand_vals = [s for t in per_task for s in per_task[t]["random_floor"].values()]
    random_finite = all(np.isfinite(s) for s in rand_vals)
    max_random_dev = max(abs(per_task[t]["baseline"] - s)
                         for t in per_task for s in per_task[t]["random_floor"].values())
    random_ok = bool(random_finite and max_random_dev <= RAND_BAND)
    controls_ok = bool(all_ablate_ok and n4_ok and random_ok)

    # freeze the instrument INTO projector_svd.pt — includes controls_ok (the unblind gate),
    # the item-hash + split alignment data, and the pt↔mix scoping disclosure.
    torch.save({"U": U.cpu(), "S": S.cpu(), "hbar": hbar.cpu(), "W_shape": list(W.shape),
                "out_dim": out_dim, "rank": r, "k_grid": list(PA.K_GRID),
                "split_seed": SPLIT_SEED, "load_seed": LOAD_SEED, "n_total": args.n_total,
                "mix_repo": mix_repo, "controls_ok": controls_ok,
                "item_hashes": item_hashes, "splits": splits,
                "proj_pt_mix_rel_frob": proj_pt_mix_rel_frob}, out / "projector_svd.pt")

    result = {
        "torch": torch.__version__, "mix_repo": mix_repo,
        "W_proj_shape": list(W.shape), "rank": r, "out_dim": out_dim,
        "singular_values_top16": [float(x) for x in S[:16]],
        "k_grid": list(PA.K_GRID), "n_total": args.n_total,
        "split_seed": SPLIT_SEED, "load_seed": LOAD_SEED,
        "projector_pt_mix_rel_frob": proj_pt_mix_rel_frob,
        "scope_note": ("Leg B characterizes the PG2-MIX connector (the only task-ready checkpoint); "
                       "the pt→mix projector drift above quantifies the gap to the pt lineage subject."),
        "coco_ci_note": ("COCO CIDEr CI is resampled-IDF (descriptive only, not a valid CIDEr-D CI) "
                         "and COCO is EXCLUDED from the concentration vote (corpus, no per-item bootstrap)."),
        "per_task": per_task,
        "checks": {"all_ablate_drop_per_item": dict(zip(per_item_tasks, all_drop)),
                   "all_ablate_ok": all_ablate_ok,
                   "n4_drop_per_item": dict(zip(per_item_tasks, n4_drop)), "n4_ok": n4_ok,
                   "random_ok": random_ok, "max_random_dev": max_random_dev, "rand_band": RAND_BAND},
        "CONTROLS_OK": controls_ok,
    }
    (out / "legB_controls.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"[W11-B-CTRL] CONTROLS_OK={controls_ok} (all_ablate_ok={all_ablate_ok} "
          f"n4_ok={n4_ok} random_ok={random_ok})")
    if not controls_ok:
        raise RuntimeError(
            f"Leg-B CONTROLS HALT: all_ablate_ok={all_ablate_ok} n4_ok={n4_ok} random_ok={random_ok}. "
            f"all_drop={all_drop} n4_drop={n4_drop}. See {out / 'legB_controls.json'}.")
    print("[W11-B-CTRL] instrument verified → SVD/h̄ frozen; concentration decision read at unblind.")


if __name__ == "__main__":
    main()
