"""Answer-token R_KL bridge — the pre-registered secondary readout (capability-readout prereg §3,
§8, Fig.4 `fig_answertoken_kl`). Completes the analysis flagged MAJOR by the adversarial audit: the
headline compares E4's R_KL (teacher-forced KL on PROMPT tokens) to R_cap (greedy ANSWER-token
generation) — different token populations. This computes R_KL on the GROUND-TRUTH ANSWER tokens, on
the SAME frozen edit family + manifest as the sweep, so the KL→capability link is tested on identical
tokens.

For each config, teacher-force (prompt + GT-answer suffix) and measure
  KL( P_PG2-pt  ||  P_edited )  averaged over the answer-predicting positions (logits[:-1] whose
  next token is a supervised answer token, labels[1:] != -100),
then R_KL_ans(k) = 1 − mean_KL(full‖edit_k) / mean_KL(full‖FLOOR)  — the exact parallel of E4's R_KL
and of R_cap. CEILING(=full pt LM) ⇒ KL≈0 ⇒ R_KL_ans=1 (anchor); FLOOR ⇒ R_KL_ans=0 (by def).

Reads the SAME no-retrain rank-k SVD edit family as the frozen sweep (reuses `_build_edit_family`).
Teacher-forced (1 forward/batch, no generation) → fast. GPU job. hf_online_setup() first. No
try/except.

    pg run infra.aws_pg.jobs.e4_capability_answerkl --args="--n=2000" \
        --type g6e.8xlarge,p4d.24xlarge --s3-prefix results/e4_capreadout_answerkl --budget-min 120
"""
from __future__ import annotations

import argparse
import json
import pathlib

from infra.aws_pg.jobs.e4_capability import _build_edit_family, _eval_rows, SEED, K_SWEEP, TASKS_ALL
from infra.aws_pg.jobs.e4_compress_curve import _set_decoder

KL_MICRO = 4
N_BOOT = 2000
N_DEFAULT = 2000


def _answer_batches(proc, task, rows, device, batch):
    """Teacher-forced (prompt + GT-answer suffix) batches with the supervised-answer mask. Uses the
    processor's `suffix=` path (same as battery_tasks training collate) → `labels` (-100 on
    image/prompt, real ids on the answer suffix)."""
    import torch
    out = []
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        images = [task.image_fn(r).convert("RGB") for r in chunk]
        prompts = [task.prompt_fn(r) for r in chunk]
        suffixes = [task.target_fn(r) for r in chunk]
        enc = proc(text=prompts, images=images, suffix=suffixes, return_tensors="pt", padding="longest")
        enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        assert "labels" in enc, "processor did not return labels for the suffix path"
        out.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()})
    return out


def _answer_kl_per_pos(model_full, model_edit, batches):
    """Per-answer-position KL(P_full || P_edit). Answer-predicting positions = logits[:, :-1] whose
    next token labels[:, 1:] is a supervised answer token (!= -100). Returns a 1-D CPU numpy array."""
    import numpy as np
    import torch
    kls = []
    for b in batches:
        ii, am, pv, labels = b["input_ids"], b["attention_mask"], b["pixel_values"], b["labels"]
        with torch.inference_mode():
            lf = model_full(input_ids=ii, attention_mask=am, pixel_values=pv).logits[:, :-1]
            le = model_edit(input_ids=ii, attention_mask=am, pixel_values=pv).logits[:, :-1]
        mask = (labels[:, 1:] != -100)                          # (b, T-1) answer-predicting positions
        pf = torch.log_softmax(lf.float(), dim=-1)[mask]        # (npos, V)
        pe = torch.log_softmax(le.float(), dim=-1)[mask]
        kl = (pf.exp() * (pf - pe)).sum(-1)                     # (npos,)
        kls.append(kl.detach().cpu().numpy())
        del lf, le, pf, pe, kl
        torch.cuda.empty_cache()
    return np.concatenate(kls) if kls else np.zeros(0)


def _r_kl_ci(kl_floor, kl_cfg, n_boot, seed):
    """Bootstrap CI for R_KL_ans = 1 - mean(kl_cfg)/mean(kl_floor), paired over answer positions."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = min(kl_floor.size, kl_cfg.size)
    point = 1.0 - (kl_cfg.mean() / kl_floor.mean()) if kl_floor.mean() > 0 else float("nan")
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        mf = kl_floor[idx].mean()
        if mf > 0:
            draws.append(1.0 - kl_cfg[idx].mean() / mf)
    return {"R_KL_ans": float(point),
            "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--batch", type=int, default=KL_MICRO)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from infra.pg import battery_tasks as B
    from infra.pg import diff_harvest as H

    H.hf_online_setup()
    device = "cuda"
    assert torch.cuda.is_available(), "answer-token R_KL needs a GPU"
    n = 32 if a.smoke else a.n
    task_names = ("refcoco",) if a.smoke else (TASKS_ALL if a.tasks == "all"
                                               else tuple(t.strip() for t in a.tasks.split(",")))
    print(f"[answerkl] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} n={n} "
          f"tasks={task_names}", flush=True)

    pg_after, pg_before, proc, paths, _img, gate = H.build_pair("base-pt", device)
    print(f"[answerkl] swap_gate={json.dumps(gate, default=str)}", flush=True)
    lm_edit, base_sd, after_sd, dkeys, builders, edited_params, ldesc = _build_edit_family(
        pg_after, pg_before, paths, device)        # pg_after KEPT (it is P_full, the KL reference)
    print(f"[answerkl] {len(builders)} configs: {[b[0] for b in builders]}", flush=True)

    results = {"experiment": "e4_capability_answerkl", "status": "confirmatory-secondary", "seed": SEED,
               "config": {"k_sweep": list(K_SWEEP), "n": n, "n_boot": N_BOOT},
               "provenance": {"axis": "base-pt", "full_ref": "google/paligemma2-3b-pt-224 (PG2-pt LM)",
                              "swap_gate": gate, "compares_to": "E4 prompt-token R_KL "
                              "(results/e4_basept_v2) + R_cap (results/e4_capreadout_sweep)"},
               "tasks": {}}

    for tname in task_names:
        task = B.TASKS[tname]
        rows = _eval_rows(task, n, SEED)
        batches = _answer_batches(proc, task, rows, device, a.batch)
        n_ans = int(sum((b["labels"][:, 1:] != -100).sum().item() for b in batches))
        print(f"\n[answerkl] === task={tname} n={len(rows)} answer_positions={n_ans} ===", flush=True)

        # FLOOR baseline KL(full || base) on answer tokens
        _set_decoder(lm_edit, base_sd, {}, device)
        kl_floor = _answer_kl_per_pos(pg_after, pg_before, batches)
        base_mean = float(kl_floor.mean())
        assert base_mean > 0, f"{tname}: answer-token KL(full||FLOOR) <= 0 — halt"
        print(f"[answerkl] {tname} baseline answer-KL(full||base)={base_mean:.4f}", flush=True)

        cfgs = {}
        for cname, edited, override in builders:
            _set_decoder(lm_edit, override, edited, device)
            kl_c = _answer_kl_per_pos(pg_after, pg_before, batches)
            rec = _r_kl_ci(kl_floor, kl_c, N_BOOT, SEED)
            rec["kl_mean"] = float(kl_c.mean())
            cfgs[cname] = rec
            print(f"[answerkl]   {tname}/{cname:11s} answer-R_KL={rec['R_KL_ans']:.3f} "
                  f"[{rec['lo']:.3f},{rec['hi']:.3f}] kl={rec['kl_mean']:.4f}", flush=True)
        results["tasks"][tname] = {"split": task.eval_split, "n": len(rows), "answer_positions": n_ans,
                                   "baseline_answer_kl": base_mean, "configs": cfgs}

    (out / "e4_capability_answerkl.json").write_text(json.dumps(results, indent=2))
    print("E4_CAP_ANSWERKL_OK", json.dumps(
        {t: {"rank8": results["tasks"][t]["configs"]["rank8"]["R_KL_ans"],
             "CEILING": results["tasks"][t]["configs"]["CEILING"]["R_KL_ans"]}
         for t in results["tasks"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
