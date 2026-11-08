"""SAV-A PREFLIGHT (prereg §5 gate i–iii; before freeze/controls/cells).

  (1) Shell instrument validity: BASE-shell full-split `mmlu_choice_logits`
      reproduces the VLAT-cached base number within 1.0 pt (pointer:
      results/vlat/metrics.json, key resolved at runtime — recomputed never
      transcribed); gemma-1 letter ids single-token.
  (2) Extraction gates: PG1 / pi0 / pi05 decoders load strict-clean into the
      shell (swap gates inside load_swapback_model), finite capped evals,
      source SHAs pinned.
  (3) Trainer smoke on BASE-shell: r=512 zero-delta at init (capped logits
      array-equal), short train finite + decreasing, AND a realized-length
      memory probe — 3 steps on the 8 LONGEST prompts of the n=16k ladder
      subset (the W17 OOM lesson).

    python -m infra.aws_pg.jobs.sav_preflight --out <dir> --vlat-metrics <repo json> \
        [--pi05-sealed-s3 s3://...] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vlat-metrics", default="results/vlat/metrics.json")
    ap.add_argument("--vlat-key", default="checkpoints.base.mmlu_acc_pct",
                    help="exact dotted path of the cached base number (pinned; critic m8)")
    ap.add_argument("--pi05-sealed-s3", default="")
    ap.add_argument("--tol-cached", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--smoke", action="store_true")
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
    from infra.pg import vlat_swapback as SB
    from infra.pg.entities import ENTITIES
    from infra.aws_pg.jobs.vlat_retention import _extract_decoder

    assert torch.cuda.is_available()
    device = "cuda"
    n_per = 2 if args.smoke else None
    print(f"[SAV-PRE] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} smoke={args.smoke}")

    ent = ENTITIES["L0_g1"]
    tok = AutoTokenizer.from_pretrained(ent.repo_id, revision=ent.revision)
    letters = R.letter_token_ids(tok)
    print(f"[SAV-PRE][1] gemma-1 letter ids {letters} (single-token ✓)")

    # ---- (1) cached-number reproduction on the BASE shell ----
    if args.vlat_metrics.startswith("s3://"):
        vlat_local = out / "vlat_metrics.json"
        subprocess.run(["aws", "s3", "cp", args.vlat_metrics, str(vlat_local),
                        "--region", "us-east-1"], check=True)
        vm = json.loads(vlat_local.read_text())
    else:
        vm = json.loads(pathlib.Path(args.vlat_metrics).read_text())

    node = vm
    for part in args.vlat_key.split("."):
        assert isinstance(node, dict) and part in node, \
            f"VLAT key {args.vlat_key!r} missing at {part!r} (keys: {list(node)[:8]})"
        node = node[part]
    cached = float(node)
    hits = [(args.vlat_key, cached)]
    print(f"[SAV-PRE][1] VLAT cached base ({args.vlat_key}) = {cached}")
    base = AutoModelForCausalLM.from_pretrained(
        ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    mm = TE.mmlu_choice_logits(base, tok, device, n_per_subject=n_per)
    d_cached = abs(mm["acc_pct"] - cached)
    gate_1 = bool(args.smoke or d_cached <= args.tol_cached)
    print(f"[SAV-PRE][1] BASE-shell MMLU={mm['acc_pct']:.2f} vs VLAT {cached} "
          f"|Δ|={d_cached:.2f} → ok={gate_1}")

    # ---- (2) extraction gates ----
    pi05_local = ""
    if args.pi05_sealed_s3:
        pi05_local = str(out / "pi05_decoder.safetensors")
        subprocess.run(["aws", "s3", "cp", args.pi05_sealed_s3, pi05_local,
                        "--region", "us-east-1"], check=True)
    del base
    torch.cuda.empty_cache()
    ext = {}
    for name in ("PG1", "pi0", "pi05"):
        dec, sha = _extract_decoder(name, device, pi05_local)
        m, info = SB.load_swapback_model(dec, device)
        ev = m(**tok(["The capital of France is"], return_tensors="pt").to(device),
               logits_to_keep=1)
        finite = bool(torch.isfinite(ev.logits).all())
        ext[name] = {"sha": str(sha), "swap_ok": True, "finite": finite,
                     "decoder_delta": info.get("decoder_max_abs_delta_vs_base")}
        print(f"[SAV-PRE][2] {name}: sha={str(sha)[:12]} finite={finite}")
        del m, dec
        torch.cuda.empty_cache()
    gate_2 = all(v["finite"] for v in ext.values())

    # ---- (3) trainer smoke + realized-length memory probe on BASE-shell ----
    base = AutoModelForCausalLM.from_pretrained(
        ent.repo_id, revision=ent.revision, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager").eval()
    pre = TE.mmlu_choice_logits(base, tok, device, n_per_subject=2)
    arep = R.attach_adapters(base.model, rank=512, seed=1337, alpha=512.0)
    base.to(device)
    post = TE.mmlu_choice_logits(base, tok, device, n_per_subject=2)
    zero_delta = bool(post["per_item"] == pre["per_item"]
                      and np.array_equal(np.asarray(post["choice_logits"]),
                                         np.asarray(pre["choice_logits"])))
    ds = load_dataset(TE.MMLU_REPO, "all")
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ded = R.dedup_against_eval(pool, mm["per_item_hash"])
    sub16 = R.ladder_subset(ded["kept"], 64 if args.smoke else 16_000)
    ex = [R.format_qa(r["question"], r["choices"], r["answer"]) for r in sub16]
    ex_sorted = sorted(ex, key=lambda e: -len(tok.encode(e["prompt"], add_special_tokens=False)))
    longest8 = ex_sorted[:8]
    t_long = R.train_adapter_qa(base, tok, device, longest8, lr=args.lr, epochs=1,
                                batch_size=8, warmup_ratio=0.0, seed=1337,
                                max_steps=3, micro_batch=2)
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    t_small = R.train_adapter_qa(base, tok, device, ex[:64], lr=args.lr, epochs=2,
                                 batch_size=8, warmup_ratio=0.03, seed=1337, micro_batch=2)
    loss_ok = bool(np.isfinite(t_small["final_loss"])
                   and t_small["mean_loss_last_epoch"] < t_small["mean_loss_first_epoch"])
    gate_3 = bool(zero_delta and np.isfinite(t_long["final_loss"]) and loss_ok
                  and arep["trainable_params"] > 0)
    print(f"[SAV-PRE][3] zero_delta={zero_delta} longest8-3steps loss={t_long['final_loss']:.3f} "
          f"peak={mem:.1f}GiB small {t_small['mean_loss_first_epoch']:.3f}"
          f"→{t_small['mean_loss_last_epoch']:.3f} → ok={gate_3}")

    ok = bool(gate_1 and gate_2 and gate_3)
    result = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "smoke": args.smoke,
              "gate_1": {"base_acc_pct": mm["acc_pct"], "vlat_cached": cached,
                         "vlat_key": hits[0][0], "abs_delta": d_cached,
                         "letter_ids": letters, "ok": gate_1},
              "gate_2_extraction": ext, "gate_2_ok": gate_2,
              "gate_3_trainer": {"zero_delta": zero_delta, "peak_mem_gib": mem,
                                 "long_loss": t_long["final_loss"],
                                 "small": {k: t_small[k] for k in
                                           ("mean_loss_first_epoch", "mean_loss_last_epoch")},
                                 "ok": gate_3},
              "dedup": {"n_in": ded["n_in"], "n_dropped": ded["n_dropped"]},
              "PREFLIGHT_OK": ok}
    (out / "preflight.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[SAV-PRE] PREFLIGHT_OK={ok}")
    if not ok:
        raise RuntimeError("SAV PREFLIGHT HALT")


if __name__ == "__main__":
    main()
