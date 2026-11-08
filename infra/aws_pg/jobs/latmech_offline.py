"""LATMECH offline gate — the §9.2 real-data, NO-model CPU check (run before any GPU minute).

Real MMLU + the real gemma-2-2b tokenizer, but NO model forward (CPU, `pg --no-weights`). Proves
the DATA wiring the weight-free dry-run (§9.1, test_latmech.py) cannot: the degenerate-cloze filter
count, the discovery/confirmation split partition, the qa_item_hash 895-collision audit (hash is a
tripwire, NOT a join key), the ladder-subset nesting + train↔eval dedup, and the cloze
continuation-boundary invariant (prompt is a clean token prefix of prompt+continuation).

    python -m infra.aws_pg.jobs.latmech_offline --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=200, help="items for the tokenizer-boundary check")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from infra.pg import relearn as Rl
    from infra.pg import reprice_stats as RS
    from infra.pg import w16_probes as W
    from infra.pg.text_eval import MMLU_REPO, _q_block_cloze
    from infra.pg.entities import ENTITIES

    ds = load_dataset(MMLU_REPO, "all")
    test = [dict(r) for r in ds["test"]]
    rep = {"n_test": len(test)}

    # 1. degenerate cloze filter (model-free).
    deg = RS.degenerate_cloze_filter(test)
    rep["degenerate"] = {"n_in": deg["n_in"], "n_dropped": deg["n_dropped"]}

    # 2. qa_item_hash collision audit (hash = sha1(question[:64])[:12]; collisions expected).
    hashes = [Rl.qa_item_hash(r["question"]) for r in test]
    counts = Counter(hashes)
    n_colliding = sum(c for c in counts.values() if c > 1)
    rep["hash_audit"] = {"n_unique": len(counts), "n_colliding_items": n_colliding,
                         "max_bucket": max(counts.values())}

    # 3. discovery/confirmation/grid split partition (disjoint + cover; the firewall).
    disc = [h for h in hashes if W.is_discovery(h)]
    conf = [h for h in hashes if W.is_confirmation(h)]
    grid = [h for h in hashes if W.is_grid_subset(h)]
    rep["split"] = {
        "n_discovery": len(disc), "n_confirmation": len(conf), "n_grid_subset": len(grid),
        "disjoint": all(not (W.is_discovery(h) and W.is_confirmation(h)) for h in set(hashes)),
        "covers": all(W.is_discovery(h) or W.is_confirmation(h) for h in set(hashes)),
    }
    assert rep["split"]["disjoint"], "discovery/confirmation split is NOT disjoint — firewall broken"
    assert rep["split"]["covers"], "discovery/confirmation split does NOT cover all items"

    # 4. ladder nesting + train↔eval dedup (nested subsets, eval items removed from train).
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ev_hashes = [Rl.qa_item_hash(r["question"]) for r in test]
    ded = Rl.dedup_against_eval(pool, ev_hashes)
    sub1 = Rl.ladder_subset(ded["kept"], 1000)
    sub2 = Rl.ladder_subset(ded["kept"], 4000)
    q1 = {Rl.qa_item_hash(r["question"]) for r in sub1}
    q2 = {Rl.qa_item_hash(r["question"]) for r in sub2}
    rep["ladder"] = {
        "n_dedup_in": ded["n_in"], "n_dedup_dropped": ded["n_dropped"],
        "n_1k": len(sub1), "n_4k": len(sub2), "nested_1k_in_4k": q1.issubset(q2),
        "train_eval_disjoint": q2.isdisjoint(set(ev_hashes)),
    }
    assert rep["ladder"]["nested_1k_in_4k"], "ladder subsets are NOT nested (1k ⊄ 4k)"
    assert rep["ladder"]["train_eval_disjoint"], "train ladder overlaps the eval set (contamination)"

    # 5. cloze continuation-boundary invariant (text-only tokenizer; prompt is a clean prefix).
    L0 = ENTITIES["L0"]
    tok = AutoTokenizer.from_pretrained(L0.repo_id, revision=L0.revision)
    n_clean = 0
    checked = 0
    for r in test[:args.sample]:
        prompt = _q_block_cloze(r["question"])
        cont = str(r["choices"][int(r["answer"])]).strip()
        if cont == "":
            continue
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        fid = tok(prompt + " " + cont, add_special_tokens=False)["input_ids"]
        checked += 1
        if fid[:len(pid)] == pid and len(fid) > len(pid):
            n_clean += 1
    rep["boundary"] = {"checked": checked, "clean_prefix": n_clean,
                       "clean_frac": round(n_clean / max(1, checked), 4)}
    assert rep["boundary"]["clean_frac"] >= 0.95, \
        f"cloze prompt is not a clean token prefix often enough: {rep['boundary']}"

    rep["OFFLINE_OK"] = True
    (out / "offline.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[LATMECH-OFFLINE] OK — {rep['split']} ladder={rep['ladder']} boundary={rep['boundary']}")


if __name__ == "__main__":
    main()
