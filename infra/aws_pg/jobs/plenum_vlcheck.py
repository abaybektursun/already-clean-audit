"""Validate `plenum_a3.vl_batch` against the REAL processor and REAL VQAv2 rows — on CPU.

Written after the VL co-training stream was found to be supervising a trailing newline instead of
the answer (PaliGemma's processor appends "\\n", so masking `[:n_prompt]` ate the first answer token;
for a single-token answer nothing but the newline survived, VL loss fell 17.04 -> 0.0045, and the
§2.3 causal intervention was applying almost no gradient).

The defect was invisible to `py_compile`, invisible to the feasibility dry-run — which only checks
that the loss is FINITE, and 0.0045 is finite — and only became visible as a suspiciously good
number 1024 training steps in. What would have caught it is a check on WHICH POSITIONS carry the
supervision, run against the real tokenizer before any GPU time. That is this job.

No policy, no GPU, no forward: the processor and the item loader are the whole dependency.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.plenum_vlcheck --type r7i.2xlarge --no-weights \\
      --args "--n 24" --s3-prefix results/plenum_vlcheck --budget-min 40
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24, help="VQAv2 rows to check")
    ap.add_argument("--batch-sizes", default="1,2,4")
    ap.add_argument("--prior-n", type=int, default=4000,
                    help="rows for the answer-prior reference. MUST match the training pool "
                         "(plenum_a3 loads 4000): a marginal estimated on the 24-row label-check "
                         "subset would be noise, not a reference level.")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    from transformers import AutoProcessor
    from infra.aws_pg.jobs.plenum_a3 import vl_batch
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_core as PC

    rep = {"object": "plenum_vlcheck_v1", "n": a.n,
           "what": "which positions carry VL supervision, against the real processor"}
    proc = AutoProcessor.from_pretrained(LS.PG1_REPO)
    tok = proc.tokenizer
    tasks = LS.mm_tasks()
    rows, _h = IP.load_items(tasks["vqav2"], a.n)
    rep["n_rows_loaded"] = len(rows)

    checks = []
    rep["checks"] = checks
    for bs in [int(x) for x in a.batch_sizes.split(",")]:
        for off in range(0, min(len(rows), 3 * bs), bs):
            batch = rows[off:off + bs]
            if len(batch) < bs:
                continue
            enc, labels = vl_batch(proc, tasks, batch, "cpu")
            n_sup = int((labels != -100).sum())
            want = 0
            decoded = []
            for r in batch:
                g, _b = PC.gold_and_banned("vqav2", r)
                ids = tok(g, add_special_tokens=False).input_ids
                want += len(ids)
                decoded.append({"answer": g, "n_answer_tokens": len(ids)})
            # THE SUFFIX API APPENDS EOS. `proc(..., suffix=answers)` supervises the answer tokens AND
            # a terminating EOS, so a 1-token answer yields 2 supervised positions. That is correct and
            # wanted — supervising EOS is what teaches the model to stop — so the expectation is
            # answer tokens PLUS one EOS per row, not exact equality. (This check failed on strict
            # equality the first time it ran against the rebuilt vl_batch, which is the check working:
            # it caught a real change in label semantics rather than waving it through.)
            assert n_sup == want + len(batch), (
                f"bs={bs} off={off}: {n_sup} supervised vs {want} answer tokens + {len(batch)} EOS")

            # THE LEAK ASSERTION, mirrored here: every supervised position must be in the CAUSAL
            # SUFFIX. A supervised position with token_type_ids == 0 sits in the bidirectional prefix,
            # where the model can attend to the answer while predicting it — the defect that drove the
            # VL loss to 0.012 and would have manufactured `U3c ≈ U3`.
            assert "token_type_ids" in enc, "no token_type_ids — the prefix/suffix split is lost"
            tti = enc["token_type_ids"]
            sup = labels != -100
            assert bool((tti[sup] == 1).all()), (
                f"bs={bs} off={off}: {int((tti[sup] != 1).sum())} supervised positions are in the "
                f"bidirectional PREFIX — the answer would be visible while being predicted")

            # And the supervised text must BE the answers (special tokens stripped, since EOS is now
            # deliberately part of the supervision).
            sup_text = []
            for i in range(labels.shape[0]):
                keep = labels[i][labels[i] != -100]
                sup_text.append(tok.decode(keep, skip_special_tokens=True).strip())
            golds = [d["answer"].strip() for d in decoded]
            assert sup_text == golds, (
                f"bs={bs} off={off}: supervised text {sup_text} != answers {golds} — the labels are "
                f"on the wrong positions")
            assert all(t not in ("", "\n") for t in sup_text), (
                f"bs={bs} off={off}: supervision decoded to empty/newline {sup_text} — the collapse "
                f"that made the co-training stream inert")
            checks.append({"batch_size": bs, "offset": off, "n_supervised": n_sup,
                           "n_expected": want, "supervised_text": sup_text, "rows": decoded})
            (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
            print(f"[VLCHK] bs={bs} off={off} n_sup={n_sup}=={want} text={sup_text}", flush=True)

    # ---- THE INTERPRETIVE FLOOR FOR THE VL LOSS -------------------------------------------
    # A falling VL loss is not by itself evidence that the co-training stream teaches anything about
    # the faculty. VQAv2 has a heavy answer prior (yes/no dominates), so a model that learns only the
    # MARGINAL distribution of answers drops the loss a long way while learning no content. Without a
    # reference level there is no way to tell "learned the prior" from "learned the task", and the
    # difference decides whether U3c's intervention has any strength behind it.
    #
    # So: the cross-entropy of always predicting the empirical answer-token marginal. A live VL loss
    # near this number means the stream is teaching the prior; meaningfully below it means content.
    # (And a loss near ZERO means a degenerate target — the newline collapse that started all this.)
    import collections
    import math
    # The prior must be estimated on the SAME pool the co-training samples from (plenum_a3 loads
    # 4000 VQAv2 rows), not on the small subset used for the label check.
    prior_rows, _ph = IP.load_items(tasks["vqav2"], a.prior_n)
    first_tok, all_toks = collections.Counter(), collections.Counter()
    n_first = n_all = 0
    for r in prior_rows:
        g, _b = PC.gold_and_banned("vqav2", r)
        ids = tok(" " + g, add_special_tokens=False).input_ids
        if not ids:
            continue
        first_tok[ids[0]] += 1; n_first += 1
        for t in ids:
            all_toks[t] += 1; n_all += 1
    ce_first = -sum(c / n_first * math.log(c / n_first) for c in first_tok.values())
    ce_all = -sum(c / n_all * math.log(c / n_all) for c in all_toks.values())
    rep["answer_prior"] = {
        "ce_first_token_nats": ce_first, "ce_all_tokens_nats": ce_all,
        "n_rows": n_first, "n_distinct_first_tokens": len(first_tok),
        "pool_matches_training": a.prior_n == 4000,
        "top_first_tokens": [[tok.decode([t]), c] for t, c in first_tok.most_common(6)],
        "how_to_read": "a live VL loss NEAR ce_all_tokens means the stream is teaching the answer "
                       "marginal and little else; meaningfully BELOW it means content; near ZERO "
                       "means a degenerate target (the trailing-newline collapse)"}
    print(f"[VLCHK] answer-prior CE: first-token {ce_first:.4f} nats, all-token {ce_all:.4f} nats "
          f"over {n_first} rows, {len(first_tok)} distinct first tokens", flush=True)
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    assert checks, "no batch was checked — the check itself would be vacuous"
    rep["verdict"] = ("PASS — VL supervision is exactly the answer tokens (+EOS) at every batch size "
                      "checked, and every supervised position is in the causal suffix")
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"PLENUM_VLCHECK_OK batches={len(checks)}", flush=True)


if __name__ == "__main__":
    main()
