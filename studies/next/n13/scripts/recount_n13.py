"""N13 recount for the independent reader (studies/next/07_PICKS/N13_SPEC.md §9.4 S5). Laptop only.

    python -m studies.next.n13.scripts.recount_n13 --banks <dir> --verdict verdict_n13.json

For every arm on MATH-500: the accuracy from the banked scores, and again from the banked extracted strings and golds re-graded with
math_verify under the active rule; both against the verdict's accuracy. Prints one line per arm and RECOUNT_OK only when all agree.
Fail-loud: no try/except, no fallback.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from infra.pg import n13_score as S
from studies.next.n13.scripts.verdict_n13 import load_bank, present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", required=True)
    ap.add_argument("--verdict", required=True)
    a = ap.parse_args()
    V = json.loads(pathlib.Path(a.verdict).read_text())
    bad = 0
    for arm, r in V["arms"].items():
        if r.get("acc") is None or not present(a.banks, arm, "math500"):
            continue
        recs, _ = load_bank(a.banks, arm, "math500")
        banked = S.acc([q["score"] for q in recs])
        regraded = S.acc([S.grade(q["gold"], q["extracted"])["score"] for q in recs])
        ok = abs(banked - r["acc"]) < 1e-9 and abs(regraded - r["acc"]) < 1e-9
        bad += not ok
        print(f"{arm}: verdict {r['acc']:.2f}, banked {banked:.2f}, regraded {regraded:.2f} {'ok' if ok else 'MISMATCH'}")
    print("RECOUNT_OK" if bad == 0 else f"RECOUNT: {bad} mismatches")


if __name__ == "__main__":
    main()
