"""TENANT anchor gate (spec §0.2 consume-at-freeze: RE-RUN, never quote) — which FT-roster
tasks have a valid criterion on the forced-choice readout: the official FT checkpoint must
beat PG1 base by >= MARGIN with the item-paired CI excluding 0, base sane (>= 0.20).
Candidates = the (ET-trainable x fc-bindable) intersection: textvqa, vqav2, cococap, refcoco
(okvqa cannot bind — no fc distractor rule; recorded, not gated). refcoco's official
checkpoint is the SEGMENTATION FT — expected invalid, measured not assumed.

    python -m infra.aws_pg.jobs.tenant_anchor --out <dir> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

CHOICE_SEED = 20260728
N_ITEMS = 400
MARGIN = 0.05
TASKS = [("textvqa", "textvqa", "google/paligemma-3b-ft-textvqa-224"),
         ("vqav2", "vqav2", "google/paligemma-3b-ft-vqav2-224"),
         ("cococap", "coco_captions", "google/paligemma-3b-ft-cococap-224"),
         ("refcoco", "refcoco", "google/paligemma-3b-ft-refcoco-seg-224")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import os
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch

    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_core as LC
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_core as PC_fc

    assert torch.cuda.is_available()
    device = "cuda"
    n_items = 24 if a.smoke else N_ITEMS
    rep = {"n_items": n_items, "margin": MARGIN, "choice_seed": CHOICE_SEED,
           "okvqa": "cannot bind — no fc distractor rule (recorded)", "tasks": {}}
    banks = {}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
        (out / "anchor_banks.json").write_text(json.dumps(banks, default=str))

    tasks = LS.mm_tasks()
    model, proc = LS.load_pg1(device=device)

    def fc_eval_batch(m, task_obj, rows, choices):
        # unscorable items -> NaN-in-place (the plenum_core contract: count and skip, keep
        # item alignment for pairing); paired stats run on the both-scorable intersection
        per = []
        for r, (cands, gold) in zip(rows, choices):
            img = task_obj.image_fn(r).convert("RGB")
            s, _ = PC_fc.score_item_fc(m, proc, device, img,
                                       PC_fc.fc_prompt(task_obj.prompt_fn(r)), cands)
            if s is None or any(x != x for x in s):
                per.append(float("nan"))
                continue
            per.append(int(int(np.argmax(s)) == gold))
        return per

    prepared = {}
    for et_name, fc_name, repo in TASKS:
        task_obj = tasks[fc_name]
        rows, hashes = IP.load_items(task_obj, 1900 if et_name == "refcoco" else 4000)
        disc, _ = LC.parity_split(hashes)
        c_rows = [r for r, m_ in zip(rows, disc) if not m_][:n_items]
        choices = PC_fc.choices_for_task(fc_name, c_rows, CHOICE_SEED)
        prepared[et_name] = (task_obj, c_rows, choices, repo)
        base_pi = fc_eval_batch(model, task_obj, c_rows, choices)
        banks[f"{et_name}|base"] = base_pi
        n_un = int(np.isnan(np.asarray(base_pi, float)).sum())
        rep["tasks"][et_name] = {"base_acc": float(np.nanmean(base_pi)), "repo": repo,
                                 "base_unscorable": n_un}
        print(f"[ANCHOR] {et_name} base acc={np.nanmean(base_pi):.4f} unscorable={n_un} "
              f"({time.time() - t0:.0f}s)", flush=True)
        _w()
    del model
    torch.cuda.empty_cache()

    from transformers import PaliGemmaForConditionalGeneration
    rng = np.random.default_rng(20260806)
    for et_name, (task_obj, c_rows, choices, repo) in prepared.items():
        ft = PaliGemmaForConditionalGeneration.from_pretrained(
            repo, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager").eval()
        ft_pi = fc_eval_batch(ft, task_obj, c_rows, choices)
        del ft
        torch.cuda.empty_cache()
        banks[f"{et_name}|ft"] = ft_pi
        base_pi = banks[f"{et_name}|base"]
        bp, fp = np.asarray(base_pi, float), np.asarray(ft_pi, float)
        ok_mask = ~(np.isnan(bp) | np.isnan(fp))
        d = fp[ok_mask] - bp[ok_mask]
        assert d.size >= 50, f"{et_name}: too few both-scorable items ({d.size})"
        idx = rng.integers(0, d.size, size=(4000, d.size))
        boots = d[idx].mean(axis=1)
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        entry = rep["tasks"][et_name]
        entry.update({"ft_acc": float(np.nanmean(fp)), "d": float(d.mean()),
                      "ci": [lo, hi], "n_paired": int(d.size),
                      "base_sane": bool(np.nanmean(bp) >= 0.20),
                      "PASS": bool(d.mean() >= MARGIN and lo > 0
                                   and np.nanmean(bp) >= 0.20)})
        print(f"[ANCHOR] {et_name}: base={np.nanmean(bp):.4f} ft={np.nanmean(fp):.4f} "
              f"d={d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] n={d.size} PASS={entry['PASS']}",
              flush=True)
        _w()

    rep["n_pass"] = sum(1 for t in rep["tasks"].values() if t.get("PASS"))
    rep["elapsed_s"] = time.time() - t0
    _w()
    print("TENANT_ANCHOR_OK", json.dumps({k: v.get("PASS") for k, v in rep["tasks"].items()}),
          flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)     # GESTALT D7
