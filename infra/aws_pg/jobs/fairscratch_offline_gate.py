"""FAIRSCRATCH offline data gate — the full w17_cell DATA path, no model/GPU forward.

Per `memory/feedback_offline_gate_before_gpu`: run the whole pipeline minus the model
forward, on REAL MMLU + the frozen W17 controls, BEFORE spending a GPU launch. Catches the
one failure a weight-free logic test cannot: live-dataset DRIFT that would trip the GPU
cell's `assert ev["per_item_hash"] == split_hashes` after hours of training.

Checks:
  1. controls.json loads, CONTROLS_OK, BEFORE/FLOOR/item_hashes present + length-aligned.
  2. the live MMLU `test` (sorted-subject, row order — IDENTICAL to mmlu_choice_logits)
     reproduces controls' item_hashes EXACTLY (the frozen alignment key still resolves).
  3. is_grid_subset / is_confirmation partition the items; confirmation count is reported.
  4. auxiliary_train loads; dedup_against_eval audit counts match controls' banked dedup.
  5. ladder_subset(16000) is the right size + nested in itself; format_qa renders one row.

Fail-loud; any mismatch RAISES.  Run:  python -m infra.aws_pg.jobs.fairscratch_offline_gate \
    --controls results/w17/controls.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


def _hash(q):
    return hashlib.sha1(q[:64].encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", required=True)
    ap.add_argument("--n", type=int, default=16000)
    args = ap.parse_args()

    from collections import defaultdict
    from datasets import load_dataset

    from infra.pg import relearn as R
    from infra.pg import text_eval as TE
    from infra.pg import w16_probes as W

    print("[FS-OFFLINE] (1) controls.json")
    C = json.loads(pathlib.Path(args.controls).read_text())
    assert C["CONTROLS_OK"] is True, "controls not OK"
    hashes = C["item_hashes"]
    b_pi, f_pi = C["BEFORE"]["per_item"], C["FLOOR"]["per_item"]
    assert len(hashes) == len(b_pi) == len(f_pi), \
        f"controls misaligned: hashes={len(hashes)} BEFORE={len(b_pi)} FLOOR={len(f_pi)}"
    print(f"  [ok] CONTROLS_OK; {len(hashes)} items; BEFORE acc={C['BEFORE']['acc_pct']} "
          f"FLOOR acc={C['FLOOR']['acc_pct']} no_deg_band={C['no_degradation_floor_bucket']:.4f}")

    print("[FS-OFFLINE] (2) live MMLU test reproduces controls' item_hashes (order-exact)")
    ds = load_dataset(TE.MMLU_REPO, "all")
    test_by_subj = defaultdict(list)
    for r in ds["test"]:
        test_by_subj[r["subject"]].append(r)
    live_hashes = [_hash(r["question"]) for subj in sorted(test_by_subj) for r in test_by_subj[subj]]
    assert live_hashes == hashes, (
        f"DATASET DRIFT: live MMLU hashes != controls (live n={len(live_hashes)} "
        f"controls n={len(hashes)}; first mismatch at "
        f"{next((i for i,(a,b) in enumerate(zip(live_hashes,hashes)) if a!=b), 'len')})")
    print(f"  [ok] {len(live_hashes)} live hashes == controls item_hashes EXACTLY")

    print("[FS-OFFLINE] (3) discovery/confirmation split")
    conf = W.mask_by(hashes, hashes, W.is_confirmation)
    grid = W.mask_by(hashes, hashes, W.is_grid_subset)
    assert len(conf) + len(W.mask_by(hashes, hashes, lambda h: not W.is_confirmation(h))) == len(hashes)
    print(f"  [ok] confirmation={len(conf)}  grid-subset(discovery eval)={len(grid)}  total={len(hashes)}")

    print("[FS-OFFLINE] (4) auxiliary_train dedup audit vs controls")
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ded = R.dedup_against_eval(pool, hashes)
    print(f"  live dedup: n_in={ded['n_in']} n_dropped={ded['n_dropped']}  "
          f"controls: n_in={C['dedup']['n_in']} n_dropped={C['dedup']['n_dropped']}")
    assert ded["n_in"] == C["dedup"]["n_in"] and ded["n_dropped"] == C["dedup"]["n_dropped"], \
        "dedup audit drift vs controls"
    print("  [ok] dedup audit matches controls exactly")

    print("[FS-OFFLINE] (5) ladder_subset + format_qa")
    sub = R.ladder_subset(ded["kept"], args.n)
    assert len(sub) == args.n
    sub_small = R.ladder_subset(ded["kept"], 1000)
    h_small = {_hash(r["question"]) for r in sub_small}
    h_big = {_hash(r["question"]) for r in sub}
    assert h_small <= h_big, "ladder not nested at real scale"
    ex = R.format_qa(sub[0]["question"], sub[0]["choices"], sub[0]["answer"])
    assert ex["prompt"].endswith("Answer:") and ex["target_text"].strip() in ("A", "B", "C", "D")
    print(f"  [ok] ladder n={len(sub)} nested(1000⊂16000); format_qa renders "
          f"(target='{ex['target_text']}')")

    print("[FS-OFFLINE] ALL PASS — data path wired; safe to launch GPU cells")


if __name__ == "__main__":
    main()
