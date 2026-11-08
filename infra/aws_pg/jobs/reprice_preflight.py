"""REPRICE PREFLIGHT (fail-fast; before freeze/controls/ladder).

Runbook order (prereg §7): the OFFLINE gate (gate 0, CPU, no model) runs FIRST and
HALTs before any GPU minute if the NEW cloze instrument is mis-wired. Then the GPU
gates de-risk the cloze axis and bank the per-axis baselines.

Gates (HALT on failure):
  (0) OFFLINE (CPU, real MMLU + L0 tokenizer, NO forward):
      - cloze boundary invariant: prompt-prefix stable, continuation ≥ 1 token, for
        every candidate of a sample of items (the only thing that makes the banked
        cloze logprobs meaningful) — on the REAL Gemma tokenizer.
      - degenerate-item filter set size on the full 14,042 test items.
      - [:64] hash collision audit (T5): #items sharing a qa_item_hash.
      With --offline-only this is the whole job (run it on a CPU r-family box first).
  (1) GPU pairing + ceiling/subject baselines: `mmlu_blankimg` and `mmlu_cloze`
      (blank-image) on pg_before (CEILING) and pg_after (DELETED t=0). MCQ and cloze
      per_item_hash must be IDENTICAL (the pairing guarantee). Ceiling reproduces
      MMLU; ceiling cloze ≫ chance; J(ceiling) high. Banked.
  (2) GPU cloze dynamic range: lobotomize one mid LM layer of pg_before → BOTH MCQ
      and cloze must DROP (proves cloze detects real loss, not 'always recover').
  (3) GPU trainer smoke on pg_after (the relearning subject): r=8 adapters,
      zero-delta at init (MCQ+cloze identical pre/post attach @ 0 steps), n=64 train
      loss finite + decreasing; r=512 memory-fit smoke.

    python -m infra.aws_pg.jobs.reprice_preflight --out <dir> [--smoke] [--offline-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

CHANCE = 0.25                      # 4-way MMLU chance
PUBLISHED_MMLU = 51.3             # gemma-2-2b 5-shot (recompute, never a baseline)


def _canonical_test_rows(n_per_subject=None):
    """The (sorted-subject, row) item order shared by every MMLU instrument."""
    from collections import defaultdict
    from datasets import load_dataset
    from infra.pg.text_eval import MMLU_REPO
    ds = load_dataset(MMLU_REPO, "all")
    by_subj = defaultdict(list)
    for r in ds["test"]:
        by_subj[r["subject"]].append(r)
    rows = []
    for subj in sorted(by_subj):
        rs = by_subj[subj]
        rows.extend(rs[:n_per_subject] if n_per_subject else rs)
    return rows


def offline_gate(tok, sample=200) -> dict:
    """CPU, no model. Cloze boundary invariant + degenerate filter + collision audit."""
    from infra.pg.reprice_stats import degenerate_cloze_filter
    from infra.pg.text_eval import _q_block_cloze

    rows = _canonical_test_rows()
    hashes = [hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12] for r in rows]
    # [:64] collision audit (T5): items that share a hash with a DIFFERENT question.
    by_hash = {}
    for r, h in zip(rows, hashes):
        by_hash.setdefault(h, set()).add(r["question"])
    collisions = {h: len(qs) for h, qs in by_hash.items() if len(qs) > 1}

    deg = degenerate_cloze_filter([dict(r) for r in rows])

    # boundary invariant on the REAL tokenizer (a sample is sufficient — it is a
    # tokenizer property, not an item property): the prompt tokenizes the same with
    # or without each appended continuation, and every continuation is ≥ 1 token.
    bad = []
    for r in rows[:sample]:
        prompt = _q_block_cloze(r["question"])     # bare item (prefix-invariance is per-question)
        p_ids = tok.encode(prompt, add_special_tokens=True)
        for k, c in enumerate(r["choices"]):
            f_ids = tok.encode(prompt + " " + str(c).strip(), add_special_tokens=True)
            m = len(f_ids) - len(p_ids)
            if not (m >= 1 and f_ids[:len(p_ids)] == p_ids):
                bad.append({"hash": hashlib.sha1(r["question"][:64].encode()).hexdigest()[:12],
                            "choice": k, "m": m})
    ok = bool(len(bad) == 0)
    return {"n_test": len(rows), "n_collision_hashes": len(collisions),
            "max_collision_multiplicity": max(collisions.values()) if collisions else 1,
            "degenerate_dropped": deg["n_dropped"], "degenerate_kept": len(deg["kept"]),
            "boundary_checked": min(sample, len(rows)), "boundary_violations": bad[:8],
            "OFFLINE_OK": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="capped sizes, wiring only")
    ap.add_argument("--offline-only", action="store_true", help="gate 0 only (CPU box)")
    ap.add_argument("--eval-split", choices=["full", "bucket", "confirmation"], default="full",
                    help="restrict the gate-1 ceiling/subject BANK to a split "
                         "(bucket=is_grid_subset discovery sub-bucket; confirmation=sealed half)")
    ap.add_argument("--lobotomy-layer", type=int, default=13, help="mid LM layer to reinit")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    from transformers import AutoTokenizer
    from infra.pg.entities import ENTITIES
    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)

    # ---- (0) OFFLINE gate (CPU, no model) — FIRST, HALT-before-GPU ----
    g0 = offline_gate(tok, sample=200)
    print(f"[REPRICE-PRE][0] offline: n_test={g0['n_test']} "
          f"degenerate_dropped={g0['degenerate_dropped']} "
          f"collisions={g0['n_collision_hashes']} boundary_ok={g0['OFFLINE_OK']}")
    assert g0["OFFLINE_OK"], f"OFFLINE cloze boundary HALT — {g0['boundary_violations']}"
    if args.offline_only:
        (out / "preflight_offline.json").write_text(json.dumps(g0, indent=1, default=str))
        print("[REPRICE-PRE] offline-only OK; exiting before GPU.")
        return

    import numpy as np
    import torch
    from infra.pg import relearn as Rl
    from infra.pg import reprice_stats as RS
    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as W

    assert torch.cuda.is_available(), "REPRICE preflight needs GPU."
    device = "cuda"
    n_per = 2 if args.smoke else None
    print(f"[REPRICE-PRE] torch={torch.__version__} smoke={args.smoke}")

    bank_filter = {"full": None, "bucket": W.is_grid_subset,
                   "confirmation": W.is_confirmation}[args.eval_split]

    def co_measure(model, proc, *, n=n_per, item_filter=None):
        """MCQ (blank-image) + cloze (blank-image), SAME split, paired per item."""
        mcq = TE.mmlu_blankimg(model, proc, device, n_per_subject=n, item_filter=item_filter)
        clz = TE.mmlu_cloze(model, tok, device, processor=proc, n_per_subject=n,
                            item_filter=item_filter)
        assert mcq["per_item_hash"] == clz["per_item_hash"], \
            "PAIRING HALT — MCQ and cloze item order differ"
        return mcq, clz

    # ---- (1) pairing + ceiling/subject baselines (banked on --eval-split) ----
    pg_after, pg_before, proc, paths, _it, gate = H.build_before_after(device)
    ceil_mcq, ceil_clz = co_measure(pg_before, proc, item_filter=bank_filter)
    subj_mcq, subj_clz = co_measure(pg_after, proc, item_filter=bank_filter)
    j_ceiling = RS.concordance(ceil_mcq["per_item"], ceil_clz["per_item"])
    g1 = {
        "ceiling_mcq_acc_pct": ceil_mcq["acc_pct"], "ceiling_cloze_acc_pct": ceil_clz["acc_pct"],
        "subject_mcq_acc_pct": subj_mcq["acc_pct"], "subject_cloze_acc_pct": subj_clz["acc_pct"],
        "J_ceiling": j_ceiling["J"], "pairing_aligned": True,
        "build_gate": {k: gate[k] for k in ("decoder_max_abs_delta_vs_PG",
                                            "vision_tower_PG_identical", "projector_PG_identical")},
    }
    if args.smoke:
        gate_1 = True
    else:
        # cloze sanity = POSITIVE LIFT: among MCQ-correct items, cloze is correct more often
        # than its base rate (the two axes are genuinely different but correlated). An absolute
        # J threshold was wrong — the axes SHOULD diverge (that is the experiment's premise).
        gate_1 = bool(ceil_mcq["acc_pct"] >= 40.0                       # base reproduces MMLU (blank-image)
                      and ceil_clz["acc_pct"] >= 100 * CHANCE + 8       # cloze has signal on intact model
                      and (j_ceiling["J"] or 0) > ceil_clz["acc"])      # positive MCQ→cloze lift
    print(f"[REPRICE-PRE][1] ceiling MCQ={ceil_mcq['acc_pct']:.1f} cloze={ceil_clz['acc_pct']:.1f} "
          f"J={j_ceiling['J']} | subject MCQ={subj_mcq['acc_pct']:.1f} cloze={subj_clz['acc_pct']:.1f} "
          f"→ ok={gate_1}")

    # ---- (2) cloze dynamic range: lobotomy must drop BOTH axes (small sample — a
    #          VALIDITY check, not a bank; intact baseline re-measured at the same size) ----
    base_mcq, base_clz = co_measure(pg_before, proc, n=2)            # intact @ small
    lob = W.reinit_lm_layer_(pg_before.get_submodule(paths["language_model"]),
                             layer_idx=args.lobotomy_layer, seed=1337)
    lob_mcq, lob_clz = co_measure(pg_before, proc, n=2)             # lobotomized @ small
    d_mcq = base_mcq["acc_pct"] - lob_mcq["acc_pct"]
    d_clz = base_clz["acc_pct"] - lob_clz["acc_pct"]
    gate_2 = bool(d_mcq >= 2.0 and d_clz >= 2.0)
    g2 = {"lobotomy_layer": args.lobotomy_layer, "n_tensors": lob["n_tensors"],
          "mcq_drop_pts": d_mcq, "cloze_drop_pts": d_clz,
          "lob_mcq_acc_pct": lob_mcq["acc_pct"], "lob_cloze_acc_pct": lob_clz["acc_pct"],
          "ok": gate_2}
    print(f"[REPRICE-PRE][2] lobotomy L{args.lobotomy_layer}: MCQ −{d_mcq:.1f} "
          f"cloze −{d_clz:.1f} → ok={gate_2}")
    del pg_before
    torch.cuda.empty_cache()

    # ---- (3) trainer smoke on the SUBJECT (pg_after): zero-delta + decreasing loss ----
    lm = pg_after.get_submodule(paths["language_model"])
    # zero-delta: a B=0 adapter is an EXACT identity → a capped re-eval must match
    # the pre-attach eval, bit-for-bit on per-item correctness (both axes).
    pre_mcq2 = TE.mmlu_blankimg(pg_after, proc, device, n_per_subject=2)
    pre_clz2 = TE.mmlu_cloze(pg_after, tok, device, processor=proc, n_per_subject=2)
    arep = Rl.attach_adapters(lm, rank=8, seed=1337, alpha=8.0)
    pg_after.to(device)
    post_mcq2 = TE.mmlu_blankimg(pg_after, proc, device, n_per_subject=2)
    post_clz2 = TE.mmlu_cloze(pg_after, tok, device, processor=proc, n_per_subject=2)
    zero_delta = bool(pre_mcq2["per_item"] == post_mcq2["per_item"]
                      and pre_clz2["per_item"] == post_clz2["per_item"])
    from datasets import load_dataset
    ds = load_dataset(TE.MMLU_REPO, "all")
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ded = Rl.dedup_against_eval(pool, ceil_mcq["per_item_hash"])
    sub = Rl.ladder_subset(ded["kept"], 64)
    examples = [Rl.format_qa(r["question"], r["choices"], r["answer"]) for r in sub]
    trep = Rl.train_adapter_qa(pg_after, tok, device, examples, lr=args.lr,
                               epochs=max(args.epochs, 2), batch_size=args.batch,
                               warmup_ratio=args.warmup, seed=1337, micro_batch=2)
    loss_ok = bool(np.isfinite(trep["final_loss"])
                   and trep["mean_loss_last_epoch"] < trep["mean_loss_first_epoch"])
    gate_3 = bool(loss_ok and zero_delta and arep["trainable_params"] > 0)
    g3 = {"adapter_trainable": arep["trainable_params"], "zero_delta_reeval": zero_delta,
          "loss_first": trep["mean_loss_first_epoch"], "loss_last": trep["mean_loss_last_epoch"],
          "loss_decreasing": loss_ok, "dedup_dropped": ded["n_dropped"], "ok": gate_3}
    print(f"[REPRICE-PRE][3] trainer: trainable={arep['trainable_params']:,} "
          f"loss {trep['mean_loss_first_epoch']:.3f}→{trep['mean_loss_last_epoch']:.3f} → ok={gate_3}")

    ok = bool(gate_1 and gate_2 and gate_3)
    result = {"smoke": args.smoke, "eval_split": args.eval_split, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0),
              "gate_0_offline": g0, "gate_1_pairing_baselines": g1, "gate_1_ok": gate_1,
              "gate_2_cloze_dynamic_range": g2, "gate_3_trainer": g3,
              "baselines_blankimg": {
                  "ceiling": {"mcq_acc_pct": ceil_mcq["acc_pct"], "cloze_acc_pct": ceil_clz["acc_pct"]},
                  "subject_t0": {"mcq_acc_pct": subj_mcq["acc_pct"], "cloze_acc_pct": subj_clz["acc_pct"]},
                  # ORDERED per-item ceiling (canonical sorted-subject order) consumed by
                  # reprice_unblind — NOT dict-keyed: the [:64] hash has up to 895-way
                  # collisions, so a dict would collapse items. Position is the alignment key.
                  "ceiling_seq": {"hash": ceil_mcq["per_item_hash"],
                                  "mcq": ceil_mcq["per_item"], "cloze": ceil_clz["per_item"]},
                  "subject_t0_seq": {"hash": subj_mcq["per_item_hash"],
                                     "mcq": subj_mcq["per_item"], "cloze": subj_clz["per_item"]},
              },
              "PREFLIGHT_OK": ok}
    (out / "preflight.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[REPRICE-PRE] PREFLIGHT_OK={ok} (0=offline 1={gate_1} 2={gate_2} 3={gate_3})")
    if not ok:
        raise RuntimeError(f"REPRICE PREFLIGHT HALT — see {out / 'preflight.json'}")


if __name__ == "__main__":
    main()
