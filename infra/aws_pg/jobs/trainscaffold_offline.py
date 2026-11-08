"""TRAINSCAFFOLD offline gate (prereg §9.2) — real-data, NO-model CPU checks (before any GPU minute).

Proves what the weight-free dry-run cannot: (1) the full-scale (~16k) invented-knowledge battery builds
+ passes the structural validity gates (rendering-disjoint, distractor type/length balance,
answer-position balance, probe-options-match); (2) the MMLU split partition + ladder dedup (REUSED
LATMECH offline); (3) the end-to-end RF_novel + 4-way verdict pipeline on PLANTED per-item cells (each
H-* outcome), proving the unblind analysis wiring. Fail-loud.

    python -m infra.aws_pg.jobs.trainscaffold_offline --out <dir> --n-entities 4000
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-entities", type=int, default=4000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from infra.pg import trainscaffold as TS
    from infra.aws_pg.jobs.trainscaffold_facts import build_battery, structural_checks

    rep = {}

    # 1. full-scale battery + structural validity.
    bat = build_battery(a.n_entities, seed=1337)
    rep["battery"] = {"n_facts": bat["meta"]["n_facts"], "hash": bat["hash"],
                      **structural_checks(bat)}
    assert bat["meta"]["n_facts"] >= 12000, f"battery too small: {bat['meta']['n_facts']}"

    # 2. MMLU split + ladder dedup (REUSED LATMECH offline; data wiring for C/REACHABILITY).
    from infra.pg import diff_harvest as H
    H.hf_online_setup()
    from datasets import load_dataset
    from infra.pg import relearn as Rl
    from infra.pg import w16_probes as W
    from infra.pg.text_eval import MMLU_REPO
    ds = load_dataset(MMLU_REPO, "all")
    hashes = [Rl.qa_item_hash(r["question"]) for r in ds["test"]]
    disc = sum(W.is_discovery(h) for h in hashes)
    conf = sum(W.is_confirmation(h) for h in hashes)
    pool = [dict(r) for r in ds["auxiliary_train"]]
    ded = Rl.dedup_against_eval(pool, hashes)
    sub16 = Rl.ladder_subset(ded["kept"], 16000)
    q16 = {Rl.qa_item_hash(r["question"]) for r in sub16}
    rep["mmlu"] = {"n_discovery": disc, "n_confirmation": conf, "n_test": len(hashes),
                   "disjoint": all(not (W.is_discovery(h) and W.is_confirmation(h)) for h in set(hashes)),
                   "n_16k": len(sub16), "train_eval_disjoint": q16.isdisjoint(set(hashes))}
    assert rep["mmlu"]["disjoint"] and rep["mmlu"]["train_eval_disjoint"], "MMLU split/dedup broken"

    # 3. end-to-end RF_novel + verdict pipeline on PLANTED cells (proves the unblind wiring).
    def planted(mean, n=2000):
        x = np.zeros(n); x[: int(round(mean * n))] = 1.0
        rng = np.random.default_rng(int(mean * 1000)); rng.shuffle(x); return x
    accR, accB = planted(0.27), planted(0.80)
    # H-capability-ish: A generalizes both formats, shuffle low. (verdict gated on reachability)
    accA_gen = planted(0.78)
    rfA = TS.rf_novel_joint_boot(accA_gen, accR, accB, n_boot=2000, seed=1)
    rf_shuf = TS.rf_novel_joint_boot(planted(0.28), accR, accB, n_boot=2000, seed=1)["point"]
    st_gen = TS.cell_state(rfA, rf_shuf)
    # H-plasticity: A at floor
    accA_floor = planted(0.28)
    st_floor = TS.cell_state(TS.rf_novel_joint_boot(accA_floor, accR, accB, n_boot=2000, seed=1), 0.0)
    base = {"R_free": "AT_FLOOR", "R_letter": "AT_FLOOR", "B_free": "GENERALIZES", "B_letter": "GENERALIZES"}
    v_cap = TS.trainscaffold_verdict({**base, "A_free": st_gen, "A_letter": st_gen}, True, False, "H-severance", 3)
    v_plast = TS.trainscaffold_verdict({**base, "A_free": st_floor, "A_letter": st_floor}, True, False, "H-capacity", 3)
    rep["pipeline"] = {"st_gen": st_gen, "st_floor": st_floor,
                       "v_capability": v_cap["named"], "v_plasticity": v_plast["named"],
                       "go_nogo": TS.go_nogo(accB, accR)["go"]}
    assert st_gen == "GENERALIZES" and st_floor == "AT_FLOOR", f"planted states wrong: {st_gen}/{st_floor}"
    assert v_cap["named"] == "H-capability" and v_plast["named"] == "H-plasticity-loss", \
        f"verdict pipeline wrong: {v_cap['named']} / {v_plast['named']}"
    assert rep["pipeline"]["go_nogo"], "planted GO/NO-GO should pass"

    rep["OFFLINE_OK"] = True
    (out / "offline.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[TS-OFFLINE] OK — battery {rep['battery']['n_facts']} facts, MMLU {rep['mmlu']}, "
          f"pipeline {rep['pipeline']}")
    print("TS_OFFLINE_OK")


if __name__ == "__main__":
    main()
