"""read-from-passage control — weight-free dry-run (prereg §9.1 / §13 gate 3).

Unit-tests the PURE logic of the read-from-passage control on tiny planted inputs — NO model, NO GPU,
seconds on a laptop. Distinct from the runtime preflight (§9.2, WIRING on real weights); this proves the
LOGIC. Fail-loud: any mismatch RAISES. Run:  python -m infra.aws_pg.jobs.rfp_dryrun
"""
from __future__ import annotations

import numpy as np

from infra.pg import read_from_passage as RFP
from infra.pg.read_from_passage import _block_cloze, _block_mcq, build_synth, item_hash
from infra.aws_pg.jobs.read_from_passage_control import reader_verdict
# reused PCR core (also covered by pcr_dryrun; re-exercised here to prove the import wiring)
from infra.aws_pg.jobs.policy_content_readout import (
    cloze_acc_lambda, label_shuffle_acc, paired_item_bootstrap, split_masks,
)


def t_generator():
    """build_synth: 4 distinct choices, gold in range AND stated in the passage, exemplar/item entity
    pools disjoint, and full determinism (same seed → identical content hash)."""
    items, exemplars, meta = build_synth(seed=20260620, n=400)
    assert len(items) == 400 and len(exemplars) == 5
    for it in items:
        assert len(it["choices"]) == 4 and len(set(it["choices"])) == 4, "choices not 4-distinct"
        assert 0 <= it["answer"] < 4
        assert it["choices"][it["answer"]] in it["passage"], "gold not stated in passage"
        # the distractor entity's value is also a choice → binding required, not keyword spotting
        assert it["subject"] in it["question"] and it["distractor_entity"] in it["passage"]
        assert it["distractor_entity"] not in it["question"], "distractor leaked into the question"
    ex_names = {e["subject"] for e in exemplars} | {e["distractor_entity"] for e in exemplars}
    it_names = {it["subject"] for it in items} | {it["distractor_entity"] for it in items}
    assert not (ex_names & it_names), "exemplar∩item entity leak"
    items2, _, meta2 = build_synth(seed=20260620, n=400)
    assert meta["content_sha1"] == meta2["content_sha1"], "generator not deterministic"
    items3, _, meta3 = build_synth(seed=20260621, n=400)
    assert meta["content_sha1"] != meta3["content_sha1"], "different seed gave identical set"
    print(f"  [ok] build_synth: 4-distinct choices, gold stated, disjoint exemplars, deterministic "
          f"(sha1={meta['content_sha1'][:8]})")


def t_item_hash():
    """item_hash is a well-distributed 12-hex key (position-pairing tripwire); distinct items differ."""
    items, _, _ = build_synth(seed=7, n=300)
    hashes = [item_hash(it) for it in items]
    assert all(len(h) == 12 and int(h, 16) >= 0 for h in hashes), "bad hash format"
    # near-unique on 300 items (collisions on a 48-bit space are vanishingly unlikely)
    assert len(set(hashes)) >= 299, f"unexpected hash collisions: {len(set(hashes))}/300"
    disc, conf = split_masks(hashes)
    assert all(d ^ c for d, c in zip(disc, conf)), "split not a partition"
    assert 100 < sum(disc) < 200, f"split imbalanced: {sum(disc)}/300"
    print(f"  [ok] item_hash: 12-hex, near-unique; split_masks clean partition (disc={sum(disc)} conf={sum(conf)})")


def t_blocks():
    """_block_cloze / _block_mcq render the passage; drop_passage removes EXACTLY the passage text and
    nothing else; cloze ends at 'Answer:' (no trailing space) so '+ \" \" + choice' is a clean append."""
    it = {"passage": "Marla is a baker. Marla's favorite color is amber.",
          "question": "What is Marla's favorite color?", "choices": ["amber", "teal", "olive", "slate"],
          "answer": 0}
    cz = _block_cloze(it["passage"], it["question"], None)
    assert cz.endswith("Answer:") and not cz.endswith("Answer: "), "cloze must end exactly at 'Answer:'"
    assert it["passage"] in cz and it["question"] in cz
    czn = _block_cloze(it["passage"], it["question"], None, drop_passage=True)
    assert it["passage"] not in czn and it["question"] in czn, "drop_passage must remove only the passage"
    assert czn == cz.replace(it["passage"].strip() + "\n", ""), "drop_passage changed more than the passage"
    mc = _block_mcq(it["passage"], it["question"], it["choices"], it["answer"])
    assert "A. amber" in mc and "B. teal" in mc and mc.rstrip().endswith("Answer: A")
    mcs = _block_mcq(it["passage"], it["question"], it["choices"], None)
    assert mcs.endswith("Answer:") and it["passage"] in mcs
    print("  [ok] blocks: passage rendered; cloze ends at 'Answer:'; drop_passage removes only the passage")


def t_reader_verdict():
    """reader_verdict resolves all four pre-registered scenarios from planted summaries (floor=0.25,
    ceiling=0.92, generic-damage null=0.27)."""
    floor, ceiling, mdw = 0.25, 0.92, 0.27
    # H_reader-intact: rfp high, CI lo above floor+null, closer to ceiling
    v = reader_verdict("pi0", 0.88, [0.88, 0.85, 0.91], floor, ceiling, mdw, mmlu_at_floor=True)
    assert v["verdict"].startswith("H_reader-intact"), v["verdict"]
    # H_reader-dead: rfp at floor (CI lo not above floor)
    v2 = reader_verdict("pi0", 0.26, [0.26, 0.23, 0.29], floor, ceiling, mdw)
    assert v2["verdict"].startswith("H_reader-dead"), v2["verdict"]
    # H_partial-reader: above floor+null but below midpoint (mid=0.585)
    v3 = reader_verdict("pi0", 0.45, [0.45, 0.40, 0.50], floor, ceiling, mdw)
    assert v3["verdict"].startswith("H_partial-reader"), v3["verdict"]
    # INDETERMINATE: above floor but CI lo not above the generic-damage null
    v4 = reader_verdict("pi0", 0.30, [0.30, 0.262, 0.34], floor, ceiling, mdw)
    assert v4["verdict"].startswith("INDETERMINATE"), v4["verdict"]
    print("  [ok] reader_verdict: intact / dead / partial / indeterminate all resolve from planted summaries")


def t_reuse_core():
    """The reused PCR pure core behaves through the read-from-passage import path: λ=1 cloze argmax,
    label-shuffle→chance, paired bootstrap CI brackets the analytic contrast."""
    logprobs = [[-3.0, -1.0, -2.0, -4.0], [-1.0, -3.0, -2.0, -5.0]]
    ntoks = [[3, 2, 2, 4], [1, 3, 5, 5]]
    per = cloze_acc_lambda(logprobs, ntoks, [1, 2], lam=1.0)
    assert per.tolist() == [1, 1], per.tolist()
    rng = np.random.default_rng(0)
    N = 4000
    sh, _ = label_shuffle_acc(rng.normal(size=(N, 4)).tolist(),
                              np.ones((N, 4), dtype=int).tolist(),
                              rng.integers(0, 4, N).tolist(), seed=4242)
    assert abs(sh - 0.25) < 0.02, f"label-shuffle did not collapse: {sh}"
    ceiling = (rng.random(N) < 0.9).astype(int)
    policy = (rng.random(N) < 0.88).astype(int)
    bt = paired_item_bootstrap({"ceiling": ceiling, "policy": policy},
                               [("ceiling_minus_policy", "ceiling", "policy")], n_boot=400, seed=1337)
    c = bt["contrast"]["ceiling_minus_policy"]
    assert c[1] < c[0] < c[2], f"CI does not bracket mean: {c}"
    print(f"  [ok] reused PCR core: cloze λ=1, label-shuffle→{sh:.4f}, paired bootstrap CI brackets mean")


def main():
    print("[RFP-DRYRUN] weight-free logic gate (prereg §9.1 / §13 gate 3)")
    t_generator()
    t_item_hash()
    t_blocks()
    t_reader_verdict()
    t_reuse_core()
    print("[RFP-DRYRUN] ALL PASS")


if __name__ == "__main__":
    main()
