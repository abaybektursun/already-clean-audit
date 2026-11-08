"""W17 PREFLIGHT (fail-fast; before the W17 freeze/controls/grid).

Gates (GPU; HALT on failure):
  (1) Cached-number reproduction: BEFORE + FLOOR(1337) full-split
      `mmlu_choice_logits` per-item arrays EXACTLY reproduce the frozen W16
      controls arrays (exact-match ≥ 0.999 + |Δacc| ≤ 0.1 — same stack
      determinism proved at W16; this re-proves it on THIS branch's code).
  (2) Trainer smoke on the BEFORE side (capability-intact): r=8 adapters,
      n=64 aux-train examples, the eq1-candidate recipe — loss finite and
      decreasing (last-epoch mean < first-epoch mean); adapter zero-delta at
      init proved on GPU (capped eval pre/post-attach identical at 0 steps).
  (3) Data gates: auxiliary_train loads (≥ 90k rows), dedup drop-rate < 5%,
      ladder nesting holds on the real pool, wikitext-103 train loads.

    python -m infra.aws_pg.jobs.w17_preflight --out <dir> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

W16_CONTROLS_S3 = "s3://pg-lineage-769063704696/results/w16_controls_a1/out/w16_controls.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--w16-controls-s3", default=W16_CONTROLS_S3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--smoke", action="store_true", help="capped sizes, wiring only")
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
    from infra.pg.entities import ENTITIES

    assert torch.cuda.is_available(), "W17 preflight needs GPU."
    device = "cuda"
    n_per = 2 if args.smoke else None
    print(f"[W17-PRE] torch={torch.__version__} smoke={args.smoke}")

    ctrl_local = out / "w16_controls_cached.json"
    subprocess.run(["aws", "s3", "cp", args.w16_controls_s3, str(ctrl_local),
                    "--region", "us-east-1"], check=True)
    W16C = json.loads(ctrl_local.read_text())
    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)

    # ---- (1) cached-number reproduction on THIS branch's code ----
    pg_after, pg_before, _proc, paths, _it, _gate = H.build_before_after(device)
    del pg_after
    torch.cuda.empty_cache()
    before_cl = TE.mmlu_choice_logits(pg_before, tok, device, n_per_subject=n_per)
    pg_floor = H.build_random_after(pg_before, paths, device,
                                    seed=int(W16C["FLOOR"]["decoder_seed"]))
    floor_cl = TE.mmlu_choice_logits(pg_floor, tok, device, n_per_subject=n_per)
    del pg_floor
    torch.cuda.empty_cache()
    if not args.smoke:
        g1 = {}
        for name, cl, key in (("BEFORE", before_cl, "BEFORE"), ("FLOOR", floor_cl, "FLOOR")):
            ref = W16C[key]["per_item"]
            match = float(np.mean(np.array(cl["per_item"]) == np.array(ref)))
            d = abs(cl["acc_pct"] - float(W16C[key]["acc_pct"]))
            g1[name] = {"exact_match": match, "abs_acc_delta": d,
                        "ok": bool(match >= 0.999 and d <= 0.1)}
            print(f"[W17-PRE][1] {name}: match={match:.4f} |Δ|={d:.3f} ok={g1[name]['ok']}")
        assert before_cl["per_item_hash"] == W16C["item_hashes"], "item order drift — HALT"
        gate_1 = all(v["ok"] for v in g1.values())
    else:
        g1 = {"smoke": True}
        gate_1 = True

    # ---- (2) trainer smoke on BEFORE (capability-intact side) ----
    lm = pg_before.get_submodule(paths["language_model"])
    cap_pre = TE.mmlu_choice_logits(pg_before, tok, device, n_per_subject=2)
    arep = R.attach_adapters(lm, rank=8, seed=1337)
    pg_before.to(device)
    cap_post = TE.mmlu_choice_logits(pg_before, tok, device, n_per_subject=2)
    # per-item exact equality is the primary criterion; logits compared exactly
    # (the all-zero fp32 delta casts to an exact bf16 zero; base kernels identical)
    zero_delta = bool(cap_post["per_item"] == cap_pre["per_item"]
                      and np.array_equal(np.asarray(cap_post["choice_logits"]),
                                         np.asarray(cap_pre["choice_logits"])))
    ds = load_dataset(TE.MMLU_REPO, "all")
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ded = R.dedup_against_eval(pool, cap_pre["per_item_hash"] if args.smoke else W16C["item_hashes"])
    sub = R.ladder_subset(ded["kept"], 64)
    examples = [R.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
    trep = R.train_adapter_qa(pg_before, tok, device, examples, lr=args.lr,
                              epochs=args.epochs, batch_size=args.batch,
                              warmup_ratio=args.warmup, seed=1337, micro_batch=2)
    n_b = len(R.qa_train_step_batches(examples, args.batch))
    loss_ok = bool(np.isfinite(trep["final_loss"])
                   and trep["mean_loss_last_epoch"] < trep["mean_loss_first_epoch"])
    gate_2 = bool(zero_delta and loss_ok and trep["steps"] == args.epochs * n_b
                  and arep["trainable_params"] > 0)
    print(f"[W17-PRE][2] zero_delta={zero_delta} steps={trep['steps']} "
          f"loss first→last {trep['mean_loss_first_epoch']:.4f}→{trep['mean_loss_last_epoch']:.4f} "
          f"→ ok={gate_2}")

    # ---- (2b) r=512 memory-fit smoke (spec §9 ledger item; primary budget) ----
    del pg_before
    torch.cuda.empty_cache()
    pg_after2, pg_before2, _p2, paths2, _i2, _g2 = H.build_before_after(device)
    del pg_after2
    torch.cuda.empty_cache()
    lm512 = pg_before2.get_submodule(paths2["language_model"])
    arep512 = R.attach_adapters(lm512, rank=512, seed=1337, alpha=2.0 * 512)
    pg_before2.to(device)
    t512 = R.train_adapter_qa(pg_before2, tok, device, examples[:32], lr=args.lr,
                              epochs=1, batch_size=args.batch,
                              warmup_ratio=args.warmup, seed=1337, max_steps=3,
                              micro_batch=2)
    gate_2b = bool(np.isfinite(t512["final_loss"]))
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"[W17-PRE][2b] r=512 fit: {arep512['trainable_params']:,} trainable, "
          f"3 steps loss={t512['final_loss']:.4f}, peak mem {mem:.1f} GiB → ok={gate_2b}")
    del pg_before2
    torch.cuda.empty_cache()

    # ---- (3) data gates incl. realized retain-budget sufficiency (MAJOR-3) ----
    wt = load_dataset(TE.WIKITEXT_REPO, "wikitext-103-raw-v1", split="train")
    n64 = {R.qa_item_hash(r['question']) for r in R.ladder_subset(ded["kept"], 64)}
    n256 = {R.qa_item_hash(r['question']) for r in R.ladder_subset(ded["kept"], 256)}
    drop_rate = ded["n_dropped"] / max(1, ded["n_in"])
    if args.smoke:
        budget_16k, pool_tokens, budget_ok = None, None, True
    else:
        sub16k = R.ladder_subset(ded["kept"], 16_000)
        ex16k = [R.format_qa(r["question"], r["choices"], r["answer"]) for r in sub16k]
        budget_16k = sum(len(tok.encode(e["prompt"], add_special_tokens=False)) + 1
                         for e in ex16k) * 3            # epochs=3 (recipe candidate)
        texts = [t for t in wt["text"][:600_000] if t.strip()]
        pool_tokens = sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)
        budget_ok = bool(pool_tokens >= budget_16k)
        print(f"[W17-PRE][3] retain budget: 16k-cell needs {budget_16k:,} tok; "
              f"pool(600k lines) has {pool_tokens:,} → ok={budget_ok}")
    gate_3 = bool(ded["n_in"] >= 90_000 and drop_rate < 0.05
                  and n64 <= n256 and len(wt) > 100_000 and budget_ok)
    print(f"[W17-PRE][3] aux={ded['n_in']} dropped={ded['n_dropped']} ({drop_rate:.4f}) "
          f"nested={n64 <= n256} wikitext_rows={len(wt)} → ok={gate_3}")

    ok = bool(gate_1 and gate_2 and gate_2b and gate_3)
    result = {
        "torch": torch.__version__, "smoke": args.smoke,
        "gate_1_reproduction": g1, "gate_1_ok": gate_1,
        "gate_2_trainer": {"zero_delta_at_init": zero_delta, "train": trep,
                           "adapter": {k: arep[k] for k in ("n_modules", "trainable_params")},
                           "ok": gate_2},
        "gate_2b_r512_fit": {"trainable_params": arep512["trainable_params"],
                             "final_loss": t512["final_loss"],
                             "peak_mem_gib": mem, "ok": gate_2b},
        "gate_3_data": {"aux_n": ded["n_in"], "dedup_dropped": ded["n_dropped"],
                        "drop_rate": drop_rate, "nested_ok": bool(n64 <= n256),
                        "wikitext_rows": len(wt),
                        "retain_budget_16k_tok": budget_16k,
                        "retain_pool_tok": pool_tokens, "budget_ok": budget_ok,
                        "ok": gate_3},
        "gpu": torch.cuda.get_device_name(0),
        "recipe_candidate": {"lr": args.lr, "epochs": args.epochs,
                             "batch": args.batch, "warmup": args.warmup},
        "PREFLIGHT_OK": ok,
    }
    (out / "preflight.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[W17-PRE] PREFLIGHT_OK={ok} (1={gate_1} 2={gate_2} 2b={gate_2b} 3={gate_3})")
    if not ok:
        raise RuntimeError(f"W17 PREFLIGHT HALT — see {out / 'preflight.json'}")


if __name__ == "__main__":
    main()
