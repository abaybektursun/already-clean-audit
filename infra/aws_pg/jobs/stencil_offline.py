"""STENCIL offline gate (CPU) — spec §9 steps 1–2, BEFORE the freeze / any GPU minute. Two halves:

  A. the weight-free dry-run logic gate (test_stencil_core.main) — the fixed-basis pin confines ΔW
     to span(U_policy); zero-delta identity; α=r⇒scale 1; grad-projection partition; iso-gain
     interpolation; degenerate fallback; every §6 adjudicator branch.
  B. the full pipeline MINUS model-forward on REAL data — the MMLU adaptation-subject roster + splits
     + train-example construction + dedup firewall; the 5 fixed-basis policy bases + contrast-range +
     natural-write census built at the real d=2048 on a synthetic bank; the real processor's
     letter-token resolution + format_qa round-trip; a small VQA/COCO/ppl battery-data smoke.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.stencil_offline --type r7i.4xlarge \
      --no-weights --s3-prefix results/stencil_offline --budget-min 60
"""
import argparse
import json
import os
import pathlib


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _stencil_shared as SS
    from infra.pg import stencil_core as SC

    SS.install_pg()
    rep = {"object": "stencil_offline_v1", "gpu_seconds": {"total": 0.0, "note": "CPU job"}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- A. weight-free dry-run logic gate ----------------------------------------------
    from infra.pg.tests import test_stencil_core as T
    n_pass = T.main()
    rep["dryrun"] = {"n_pass": int(n_pass), "n_total": len(T.ALL)}
    assert n_pass == len(T.ALL), "dry-run logic gate incomplete"
    _w()

    # ---- B1. MMLU adaptation roster + splits + train examples + dedup --------------------
    mmlu = SS.mmlu_by_subject()
    roster = SS.build_roster(mmlu)
    assert roster["discovery"] and roster["confirm"] and roster["broad_mmlu"], "empty roster partition"
    assert not (set(roster["adapt"]) & set(roster["broad_mmlu"])), "broad-MMLU overlaps adaptation"
    # per-subject train/eval split + dedup on the first discovery subject
    subj = roster["discovery"][0]
    train, evl, ev_hashes = SS.subject_train_eval(mmlu[subj]["test"])
    examples, dedup = SS.train_examples(train, ev_hashes)
    assert dedup["n_dropped"] == 0 or dedup["n_dropped"] == dedup["n_in"] - len(dedup["kept"]), "dedup audit"
    assert examples and all("prompt" in e and "answer_idx" in e for e in examples), "format_qa shape"
    # the contamination firewall: NO train example's hash collides with the eval half
    ev = set(ev_hashes)
    from infra.pg.relearn import qa_item_hash
    assert not any(qa_item_hash(r["question"]) in ev for r in dedup["kept"]), "train↔eval leak"
    rep["roster"] = {"discovery": roster["discovery"], "confirm": roster["confirm"],
                     "broad_mmlu": roster["broad_mmlu"], "n_eligible": roster["n_eligible"],
                     "test_counts": roster["test_counts"],
                     "example_subject": subj, "n_train_examples": len(examples),
                     "n_eval": len(evl), "dedup": dedup | {"kept": len(dedup["kept"])}}
    _w()

    # ---- B2. policy bases + contrast-range + census at the REAL d=2048 (synthetic bank) --
    d = SS.D
    g = torch.Generator().manual_seed(0)
    bank = {"Q_evecs": torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0],
            "sigma_evecs": torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0],
            "u_wr": torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0][:, :679],
            "d": d}
    basis_shapes = {}
    for arm in SC.FIXED_ARMS:
        for r in SC.RANK_LADDER:
            U = SC.build_policy_basis(arm, bank, r, seed=7)
            assert U.shape == (d, r), (arm, r, U.shape)
        basis_shapes[arm] = "ok@all_ranks"
    cr = {r: SC.contrast_range(bank, r)["overlap"] for r in SC.RANK_LADDER}
    # a random ΔW planted entirely in the energy span → census reads energy≈1 (wiring proof)
    r0 = 50
    U_en = bank["sigma_evecs"][:, :r0]
    dW = (U_en @ torch.randn(r0, 512, dtype=torch.float64)).numpy()
    cen = SC.natural_write_census(dW, bank, r0, "down_proj", seed=1)
    assert abs(cen["energy"] - 1.0) < 1e-8, cen
    rep["basis_gate"] = {"basis_shapes": basis_shapes, "contrast_range_overlap": cr,
                         "census_energy_planted": cen}
    _w()

    # ---- B3. real processor: letter-token resolution + format_qa forward-shape (no model) -
    from infra.pg import inherit_pg1 as IP
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(IP.PG1_REPO)
    from infra.pg.relearn import letter_token_ids
    letters = letter_token_ids(proc.tokenizer)
    assert len(set(letters)) == 4, letters
    ex = examples[0]
    enc = proc.tokenizer([ex["prompt"]], return_tensors="pt", add_special_tokens=True)
    assert enc["input_ids"].shape[0] == 1 and enc["input_ids"].shape[1] > 5, "prompt tokenization"
    rep["processor_gate"] = {"letter_token_ids": [int(x) for x in letters],
                             "example_prompt_len": int(enc["input_ids"].shape[1])}
    _w()

    # ---- B4. battery-data smoke (small n; confirms streaming loaders + Σ-cost surfaces) ---
    from infra.aws_pg.jobs._specter_shared import wikitext_windows
    vqa_task, coco_task = IP.pg1_tasks()
    vqa_rows, _ = IP.load_items(vqa_task, 4)
    coco_rows, _ = IP.load_items(coco_task, 4)
    wins = wikitext_windows(proc, 4)
    assert len(vqa_rows) == 4 and len(coco_rows) == 4 and len(wins) == 4, "battery loaders"
    rep["battery_smoke"] = {"vqa_rows": len(vqa_rows), "coco_rows": len(coco_rows),
                            "ppl_windows": len(wins), "vqa_task": vqa_task.name,
                            "coco_task": coco_task.name}
    _w()

    print("STENCIL_OFFLINE_OK", json.dumps({"dryrun": rep["dryrun"], "n_eligible": roster["n_eligible"],
                                            "discovery": roster["discovery"], "confirm": roster["confirm"]}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
